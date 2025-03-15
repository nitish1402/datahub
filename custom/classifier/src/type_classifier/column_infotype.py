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

from type_classifier import ColumnInfo, Metadata, InfotypeProposal, DebugInfo

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
class PredictionFactors:
    name: float = 0.0
    description: float = 0.0
    datatype: float = 0.0
    values: float = 0.0

@dataclass
class RegexConfig:
    regex: List[str] = field(default_factory=list)
    library: List[str] = field(default_factory=list)

@dataclass
class ClassificationConfig:
    prediction_factors_and_weights: PredictionFactors
    name: RegexConfig
    values: RegexConfig

class DataClassificationLabels(Enum):
    PAN = 'PAN'
    EMAIL = 'EMAIL'
    AADHAAR = 'AADHAAR'

class DataClassifier(Classifier):
    """Classify the data in each column using spaCy, extending DataHub's Classifier."""

    def __init__(self, classification_mode: str, regex_patterns: Dict[str, ClassificationConfig]):
        self.classification_mode = classification_mode
        self.regex_patterns = regex_patterns  # Now validated via a structured class
        self.nlp = self.custom_nlp()

    def custom_nlp(self):
        nlp = English()
        nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
        ruler = nlp.add_pipe("entity_ruler")

        patterns = []
        for label, config in self.regex_patterns.items():
            value_regexes = getattr(config, "values", None)
            if value_regexes:
                value_regexes = getattr(value_regexes, "regex", [])
            else:
                value_regexes = []

            logger.info(f"value regexes are ****** : {value_regexes}")

            for regex in value_regexes:
                logger.info(f"label is ****** : {label}")
                if label == "EMAIL":
                    patterns.append({"label": label, "pattern": [{"LIKE_EMAIL": True}]})
                else:
                    patterns.append({"label": label, "pattern": [{"TEXT": {"REGEX": regex}}]})

        ruler.add_patterns(patterns)
        return nlp

    def custom_nlp_old(self):
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
            logger.info(f"columns value ****** : {column}")
            column.infotype_proposals = self.extract_series_metadata(column.metadata, column.values)
        return columns

    def extract_series_metadata(self, metadata: Metadata, values: List[Any]) -> List[InfotypeProposal]:
        """Classify a column based on metadata (name, description) and values."""
        proposals = []
        column_name = metadata.name.lower() if metadata.name else ""

        values_weight = 1.0

        for label, config in self.regex_patterns.items():
            weights = config.prediction_factors_and_weights if hasattr(config, "prediction_factors_and_weights") else {}

            logger.info(f"weights are ****** : {weights}")
            name_weight = getattr(weights, "name", 0)

            # Check name-based classification
            name_regexes = getattr(config, "name", None)
            if name_regexes:
                name_regexes = getattr(name_regexes, "regex", [])
            else:
                name_regexes = []


            for regex in name_regexes:
                if re.search(regex, column_name, re.IGNORECASE):
                    logger.info(f"Name confidence is ****** : {name_weight * 100}")
                    values_weight -= name_weight
                    proposals.append(InfotypeProposal(
                        infotype=label,
                        confidence_level=name_weight * 100,  # Convert weight to percentage
                        debug_info=DebugInfo(name=name_weight)
                    ))


        # Detect based on values this should run only once
        detection_results = [self.detect_pan_email(self.aadhaar_pre_process(val)) for val in values if val]
        valid_results = [res for res in detection_results if res['type']]

        logger.info(f"result  is ****** : {valid_results}")

        if valid_results:
            avg_confidence = sum(res['confidence'] / 100 for res in valid_results) / len(valid_results)
            logger.info(f"Average confidence is ****** : {avg_confidence}")

            weighted_confidence = avg_confidence * values_weight * 100

            logger.info(f"weighed value confidence is ****** : {weighted_confidence}")
            detected_type = valid_results[0]['type']  # Assuming the first detected type is the dominant one

            proposals.append(InfotypeProposal(
                infotype=detected_type,
                confidence_level=weighted_confidence,
                debug_info=DebugInfo(values=avg_confidence)
            ))

        return proposals


    @classmethod
    def create(cls, config_dict: Dict[str, Any]) -> "DataClassifier":
        classification_mode = config_dict.get("classification_mode", "default")
        raw_patterns = config_dict.get("regex_patterns", {})

        validated_patterns = {}
        for label, pattern in raw_patterns.items():
            logger.info(f"Pattern is ****** : {pattern}")
            validated_patterns[label] = ClassificationConfig(
                prediction_factors_and_weights=PredictionFactors(**pattern.get("prediction_factors_and_weights", {})),
                name=RegexConfig(**pattern.get("name", {})),
                values=RegexConfig(**pattern.get("values", {}))
            )

        logger.info(f"Validated pattern is  is ****** : {validated_patterns}")
        classifier = cls(classification_mode, validated_patterns)
        return classifier

    def detect_pan_email(self, val):
        result = {'type': None, 'confidence': 0.0}
        if val is None:
            return result

        nlp_res = self.nlp(str(val))
        detected_labels = [entity.label_ for entity in nlp_res.ents]

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


class VerhoeffTable(Enum):
    """Multiplication and Permutation table for Verhoeff Algorithm."""
    MULTIPLICATION = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
                      [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
                      [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
                      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    PERMUTATION = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
                   [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
                   [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]

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