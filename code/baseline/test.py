from nltk.tokenize import sent_tokenize

text_1 = "This is a terrible product. I would return it if I had gotten it from a store. t cooks only on one side at a time. Takes 20 minutes to make a waffle. I have never returned anything on Amazon but if I knew how I would return this waffle maker. I will never use it again."

text_2 = "Generalizing backward propagation, using formal methods from supersymmetry."

# 提取句子
sentences_1 = sent_tokenize(text_1)
sentences_2 = sent_tokenize(text_2)

refs = []
for item in sentences_1:
    items = [item]
    refs.append(items)

print(refs)