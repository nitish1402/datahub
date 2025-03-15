import collections
import logging
from typing import List, Dict, Any

from datahub.ingestion.glossary.classifier import Classifier
from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngine, SpacyNlpEngine, NerModelConfiguration
from type_classifier import ColumnInfo, Metadata, InfotypeProposal, DebugInfo

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
)

class PIIClassifier(Classifier):
    """Classify the data in each column using spaCy, extending DataHub's Classifier."""

    def __init__(self, classification_confidence: float):
        _nlp = None

        self.classification_confidence = classification_confidence
        if not PIIClassifier._nlp:
            PIIClassifier._nlp = self.custom_nlp()
            self.nlp = PIIClassifier._nlp

    def classify(self, columns: List[ColumnInfo]) -> List[ColumnInfo]:
        """Classify a list of columns."""
        for column in columns:
            logger.info(f"columns value ****** : {column}")
            column.infotype_proposals = self.extract_series_metadata(column.metadata, column.values)
        return columns

    @classmethod
    def create(cls, config_dict: Dict[str, Any]) -> "PIIClassifier":
        classification_confidence = config_dict.get("classification_confidence", 1.0)
        classifier = cls(classification_confidence)
        return classifier


    def custom_nlp(self):
        logger.info(f"Input classification confidence is : {self.classification_confidence}")
        model_config = [{"lang_code": "en", "model_name": "en_core_web_sm"}]
        spacy_nlp_engine = SpacyNlpEngine(models= model_config)
        analyzer = AnalyzerEngine(default_score_threshold=self.classification_confidence,nlp_engine=spacy_nlp_engine)
        batch_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)

        # displaying the available definitions
        logger.info(f"Loaded entities for english are : {analyzer.get_supported_entities('en')}")
        logger.info(f"Loaded recognisers for english are : {[rec.name for rec in analyzer.registry.get_recognizers('en', all_fields=True)]}")
        logger.info(f"Loaded NER models are : {analyzer.nlp_engine}")

        return batch_analyzer


    def extract_series_metadata(self, metadata: Metadata, values: List[Any]) -> List[InfotypeProposal]:
        """Classify a column based on metadata (name, description) and values."""
        proposals = []

        # Detect based on values we will use the first classification from the available results
        detection_results = self.nlp.analyze_iterator(values, language='en')
        logger.info(f"detected result is : {detection_results} ")
        first_values = [sublist[0] for sublist in detection_results if sublist]

        grouped_map: Dict[str, List[RecognizerResult]] = collections.defaultdict(list)

        for obj in first_values:
            grouped_map[obj.entity_type].append(obj)


        for type, results in grouped_map.items():
            entity_type_confidence = 0.0
            for value in results:
                entity_type_confidence += value.score

            avg_confidence = entity_type_confidence / len(results)
            logger.info(f"detected type is : {type} and avg_confidence is = {avg_confidence}")

            proposals.append(InfotypeProposal(
                infotype=type,
                confidence_level=avg_confidence,
                debug_info=DebugInfo(values=avg_confidence)
            ))

        return proposals