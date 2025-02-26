import pandas as pd
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
from spacy.tokenizer import Tokenizer
import re
import spacy
from enum import Enum
import logging
from datahub.ingestion.glossary.classifier import Classifier
from dataclasses import dataclass
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
)

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DebugInfo:
    name: Optional[float] = None
    description: Optional[float] = None
    datatype: Optional[float] = None
    values: Optional[float] = None


@dataclass
class InfotypeProposal:
    infotype: str
    confidence_level: float
    debug_info: DebugInfo


@dataclass
class Metadata:
    meta_info: Dict[str, Any]
    name: str = field(init=False)
    description: str = field(init=False)
    datatype: str = field(init=False)
    dataset_name: str = field(init=False)

    def __post_init__(self):
        self.name = self.meta_info.get("Name", None)
        self.description = self.meta_info.get("Description", None)
        self.datatype = self.meta_info.get("Datatype", None)
        self.dataset_name = self.meta_info.get("Dataset_Name", None)


@dataclass
class ColumnInfo:
    metadata: Metadata
    values: List[Any]
    infotype_proposals: Optional[List[InfotypeProposal]] = None

class DataClassificationLabels(Enum):
    PAN = 'PAN'
    EMAIL = 'EMAIL'
    AADHAAR = 'AADHAAR'

class DataClassifier(Classifier):
    """Classify the data in each column using spaCy, extending DataHub's Classifier."""
    
    def __init__(self, classification_mode: str):
        self.classification_mode = classification_mode
        self.nlp = self.custom_nlp()
        
    def custom_nlp(self):
        nlp = English()
        nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
        ruler = nlp.add_pipe("entity_ruler")
        patterns = [
            {"label": DataClassificationLabels.PAN.value, "pattern": [{'TEXT': {'REGEX': "^[a-zA-Z]{3}[pPcChHfFatTblLjJgG][A-Za-z][\d]{4}[A-Za-z]$"}}]},
            {"label": DataClassificationLabels.AADHAAR.value, "pattern": [{'TEXT': {'REGEX': "^[2-9]{1}[0-9]{3}[0-9]{4}[0-9]{4}$"}}]},
            {"label": DataClassificationLabels.EMAIL.value, "pattern": [{"LIKE_EMAIL": True}]}
        ]
        ruler.add_patterns(patterns)
        return nlp
        
    def classify(self, columns: List[ColumnInfo]) -> List[ColumnInfo]:
        """Classify a list of columns."""
        for column in columns:
            column.infotype_proposals = self.extract_series_metadata(column.metadata, column.values)
        return columns
    
    def extract_series_metadata(self, metadata: Metadata, values: List[Any]) -> List[InfotypeProposal]:
        """Classify a column based on its metadata and values."""
        proposals = []
        column_name = metadata.name.lower() if metadata.name else ""

        # Detect based on column name
        if "email" in column_name:
            proposals.append(InfotypeProposal(
                infotype=DataClassificationLabels.EMAIL.value,
                confidence_level=100.0,
                debug_info=DebugInfo(name=1.0)
            ))
        elif "pan" in column_name:
            proposals.append(InfotypeProposal(
                infotype=DataClassificationLabels.PAN.value,
                confidence_level=90.0,
                debug_info=DebugInfo(name=0.9)
            ))
        elif "aadhar" in column_name:
            proposals.append(InfotypeProposal(
                infotype=DataClassificationLabels.AADHAAR.value,
                confidence_level=95.0,
                debug_info=DebugInfo(name=0.95)
            ))

        return proposals
    
    @classmethod
    def create(cls, config_dict: Dict[str, Any]) -> "DataClassifier":
        classification_mode = config_dict.get("classification_mode", "default")

        logger.info("coming here debug place 1111")

        if not isinstance(classification_mode, str):
            raise TypeError("Expected 'classification_mode' to be a string")

        return cls(classification_mode)
    
    # def extract_series_metadata(self, series_name, series):
    #     metadata = {'type': None, 'confidence': 0.0, 'val_type': []}
    #     for val in series:
    #         new_val = self.aadhaar_pre_process(val)
    #         res = self.detect_pan_email(new_val)
    #         metadata['type'] = res['type']
    #         metadata['confidence'] = res['confidence']
    #         metadata['val_type'].append({'type': res['type'], 'confidence': res['confidence']})
        
    #     logger.info(f"{series_name}: {metadata}")
    #     return metadata
    
    def detect_pan_email(self, val):
        result = {'type': None, 'confidence': 0.0}
        if val is None:
            return result
        
        nlp_res = self.nlp(str(val))
        detected_labels = [entity.label_ for entity in nlp_res.ents]
        logger.info(detected_labels)
        
        if not detected_labels:
            return result 
        
        if DataClassificationLabels.PAN.value in detected_labels:
            result['type'] = 'PAN'
            result['confidence'] = 100.0 if PanValidator.validate(val) else 0.0
        if DataClassificationLabels.AADHAAR.value in detected_labels:
            result['type'] = 'AADHAAR'
            result['confidence'] = 100.0 if AadhaarValidator.validate(val) else 0.0
        if DataClassificationLabels.EMAIL.value in detected_labels:
            result['type'] = 'Email'
            result['confidence'] = 100.0
        
        return result
    
    def aadhaar_pre_process(self, val):
        if re.search(r"^[2-9]{1}[0-9]{3}[-|\s][0-9]{4}[-|\s][0-9]{4}$", str(val)):
            return val.replace(' ', '').replace('-', '')
        return val
        
class PanValidator:
    @classmethod
    def validate(cls, val):
        try:
            return int(val[5:9]) != 0
        except ValueError:
            logger.info("ValueError", exc_info=True)
            return False
    
class AadhaarValidator:
    @staticmethod
    def validate(aadharNum):
        try:
            i = len(aadharNum)
            j = 0
            x = 0
            while i > 0:
                i -= 1
                x = VerhoeffTable.MULTIPLICATION.value[x][VerhoeffTable.PERMUTATION.value[(j % 8)][int(aadharNum[i])]]
                j += 1
            return x == 0
        except (ValueError, IndexError):
            logger.error("Validation Error", exc_info=True)
            return False
    
if __name__ == '__main__':
    df = pd.read_csv('./classification_data.csv')
    columns = [ColumnInfo(name=col) for col in df.columns]
    logger.info(DataClassifier(df).classify(columns))
