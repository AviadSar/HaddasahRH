import torch
import sentencepiece
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tuner007/t5_abs_qa")
model = AutoModelWithLMHead.from_pretrained("tuner007/t5_abs_qa")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

context = "The following text describes a patient: 86 years old, married to Yosef 96 years old. So far completely independent in her home. In the background, heart failure is currently worsening. Lives in her house floor 2-20 steps. Did not need any assistance. Recently slightly functional decline. An examination of the Nursing Law was recommended, and in addition, an application was made to the Holocaust Survivors' Welfare Fund for immediate domestic assistance. 055-6641013 Jewish. Bat Bracha 054-9292418 (head nurse in Hod Amal)."
question = "Where does the patient live?"

input_text = "context: %s <question for context: %s </s>" % (context, question)
features = tokenizer([input_text], return_tensors='pt')
out = model.generate(input_ids=features['input_ids'].to(device), attention_mask=features['attention_mask'].to(device))
print(tokenizer.decode(out[0]))