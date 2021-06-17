import torch
from transformers import BertModel, BertTokenizerFast, BertLMHeadModel, BertForMaskedLM

tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
alephbert = BertForMaskedLM.from_pretrained('onlplab/alephbert-base')
# alephbert = BertModel.from_pretrained('onlplab/alephbert-base')

# if not finetuning - disable dropout
alephbert.eval()

sentence = "הטקסט הבא מתאר מטופלת קשישה: בת 86 נשואה ליוסף בן 96. עד כה עצמאית לחלוטין בביתה. ברקע סובלת מאי ספיקת לב כעת בהחמרה. מתגוררת בביתה קומה 2- 20 מדרגות. לא נזקקה לסיוע כלשהו. לאחרונה מעט ירידה תפקודית. הומלץ על בחינה לחוק סיעוד ובנוסף נעשתה פניה לקרן לרווחת ניצולי שואה לצורך קבלת סיוע בייתי מיידי. 055-6641013יהודית. בת ברכה 054-9292418 (אחות ראשית בהוד עמל). האם המטופלת היא מלכת אנגליה? [MASK]"

inputs = tokenizer(sentence, return_tensors="pt")
# labels = tokenizer("מי הזיז את הגבינה שלי?", return_tensors="pt")['input_ids']

outputs = alephbert(**inputs)

logits = outputs.logits
softies = torch.softmax(logits, dim=2)

print("yes: " + str(softies[0][-2][2204]))
print("no: " + str(softies[0][-2][1813]))

# prediction for first 3 and last 2 tokens
#
# for i in range(10):
#     print("1-" + str(i) + ") " + tokenizer.decode(softies.topk(10, dim=2).indices[0][1][i]))
# print("\n=========================================================================================================\n")
# for i in range(10):
#     print("2-" + str(i) + ") " + tokenizer.decode(softies.topk(10, dim=2).indices[0][2][i]))
# print("\n=========================================================================================================\n")
# for i in range(10):
#     print("3-" + str(i) + ") " + tokenizer.decode(softies.topk(10, dim=2).indices[0][3][i]))
# print("\n=========================================================================================================\n")
# for i in range(10):
#     print("-3-" + str(i) + ") " + tokenizer.decode(softies.topk(10, dim=2).indices[0][-3][i]))
# print("\n=========================================================================================================\n")
# for i in range(10):
#     print("-2-" + str(i) + ") " + tokenizer.decode(softies.topk(10, dim=2).indices[0][-2][i]))
