# import pandas as pd
# from spacy.lang.en import English
# from spacy.pipeline import EntityRuler
# from spacy.matcher import Matcher
# from spacy.tokens import Span
# from spacy.tokenizer import Tokenizer
# import re
# import spacy
# from enum import Enum
# import logging
# from datahub.ingestion.glossary.classifier import Classifier

# logger = logging.getLogger(__name__)

# logging.basicConfig(
#     format="{asctime} - {levelname} - {message}",
#     style="{",
#     datefmt="%Y-%m-%d %H:%M",
#     level=logging.DEBUG,
# )

# class DataClassificationLabels(Enum):
#     PAN = 'PAN'
#     EMAIL = 'EMAIL'
#     AADHAAR = 'AADHAAR'

# class DataClassifier:
#     """classify the data in each columns using spacy."""
#     def __init__(self, arg):
#         super(DataClassifier, self).__init__()
#         self.df = arg
#         # self.en_nlp = spacy.load("en_core_web_trf")
#         self.nlp = self.custom_nlp()
        
#     def custom_nlp(self):
#         nlp = English()
#         nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
#         ruler = nlp.add_pipe("entity_ruler")
#         patterns = [{"label": DataClassificationLabels.PAN.value, "pattern": [{'TEXT' : {'REGEX': "^[a-zA-Z]{3}[p|P|c|C|h|H|f|F|a|A|t|T|b|B|l|L|j|J|g|G][A-Za-z][\d]{4}[A-Za-z]$"}}]},
#                     {"label": DataClassificationLabels.AADHAAR.value, "pattern": [{'TEXT' : {'REGEX': "^[2-9]{1}[0-9]{3}[0-9]{4}[0-9]{4}$"}}]},
#                     {"label": DataClassificationLabels.EMAIL.value, "pattern": [{"LIKE_EMAIL": True}]}]
#         ruler.add_patterns(patterns)
#         return nlp
        
#     def classify(self):
#         metadata = {}
#         for series_name, series in df.items():
#             metadata[series_name] = self.extract_series_metadata(series_name, series)
#         return metadata
    
#     def extract_series_metadata(self, series_name, series):
#         metadata = {}
#         metadata['column_name'] = {'type': None, 'confidence': 0.0, 'val_type': []}
#         for val in series:
#             # en_nlp_res = self.en_nlp(str(val))
#             new_val = self.aadhaar_pre_process(val)
#             res = self.detect_pan_email(new_val)
#             metadata['column_name']['type'] = res['type']
#             metadata['column_name']['confidence'] = res['confidence']
#             metadata['column_name']['val_type'].append({'type': res['type'], 'confidence': res['confidence'] })
        
#         logger.debug(series_name, metadata)
#         return metadata
    
#     def detect_pan_email(self, val):
#         result = {'type': None, 'confidence': 0.0}
#         if val is None:
#             return result
        
#         nlp_res = self.nlp(str(val))
#         detected_labels = []
#         for entity in nlp_res.ents:
#             detected_labels.append(entity.label_)
#         logger.debug(detected_labels)
#         if len(detected_labels) == 0:
#             return result 
#         if DataClassificationLabels.PAN.value in detected_labels:
#             result['type'] = 'PAN'
#             result['confidence'] = 100.0 if PanValidator.validate(val) else 0.0
#         if DataClassificationLabels.AADHAAR.value in detected_labels:
#             result['type'] = 'AADHAAR'
#             result['confidence'] = 100.0 if AadhaarValidator.validate(val) else 0.0
#         if DataClassificationLabels.EMAIL.value in detected_labels:
#             result['type'] = 'Email'
#             result['confidence'] = 100.0
#         return result
    
#     def aadhaar_pre_process(self, val):
#         if re.search(r"^[2-9]{1}[0-9]{3}[-|\s][0-9]{4}[-|\s][0-9]{4}$",str(val)):
#             n_val = val.replace(' ', '').replace('-', '')
#             # logger.debug(f"pre - {n_val=},{val=}")
#             return n_val
#         return val
        
    
# class PanValidator:
#     @classmethod
#     def validate(self, val):
#         try:
#             return int(val[5:9]) != 0
#         except ValueError:
#             logger.debug("ValueError", exc_info=True)
#             return False
    
# class VerhoeffTable(Enum):
#     """Multiplication and Permutation table for Verhoeff Algorithm."""
#     MULTIPLICATION = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
#         [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
#         [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
#         [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
#     PERMUTATION = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
#         [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
#         [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]

    
# class AadhaarValidator:
#     def validate(aadharNum):
#         try:
#             i = len(aadharNum)
#             j = 0
#             x = 0

#             while i > 0:
#                 i -= 1
#                 x = VerhoeffTable.MULTIPLICATION.value[x][VerhoeffTable.PERMUTATION.value[(j % 8)][int(aadharNum[i])]]
#                 j += 1
            
#             return x == 0
        
#         except ValueError:
#             logger.error("ValueError", exc_info=True)
#             return False
#         except IndexError:
#             logger.error("IndexError", exc_info=True)
#             return False


# if __name__ == '__main__':
#     df = pd.read_csv('./classification_data.csv')
#     logger.info(DataClassifier(df).classify())