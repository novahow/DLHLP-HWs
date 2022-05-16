from fairseq.models.transformer_lm import TransformerLanguageModel
import numpy as np
from tqdm import tqdm


model_dir = '/work/dlhlp2022/HW5/Assignment-cross-lingual-transfer/7.5B'
lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='sentencepiece')
lm = lm.eval()
lm = lm.half()
lm = lm.cuda()

# load xnli
lang_codes = list()
labels = list()
premises = list()
hypotheses = list()
for i, line in enumerate(open('/work/dlhlp2022/HW5/XNLI-1.0/xnli.test.tsv').readlines()):
    if i == 0:
        continue
    line = line.split('\t')
    lang_codes.append(line[0])
    labels.append(line[1])
    premises.append(line[6])
    hypotheses.append(line[7])


lang_codes = np.array(lang_codes)
labels = np.array(labels)
premises = np.array(premises)
hypotheses = np.array(hypotheses)

languages = ['English', 'French', 'Russian', 'Chinese', 'Hindi', 'Urdu', 'Bulgarian', 'Vietnamese']
codes = ['en', 'fr', 'ru', 'zh', 'hi', 'ur', 'bg', 'vi']
code_to_lang = dict(zip(np.unique(lang_codes), languages))

with open('./others_to_en.tsv', 'w') as f:

    for code in tqdm(codes):
        if code == 'en':
            continue

        ind = np.where(lang_codes == code)#[0][:100]
        current_premises = premises[ind]
        current_hypotheses = hypotheses[ind]
        current_labels = labels[ind]
        
        premise_examples = list()
        hypotheses_examples = list()
        min_len_b_premises = list()
        min_len_b_hypotheses = list()
        
        prompt = 'Translate from {} to English: \n'.format(code_to_lang[code])

        for premise, hypothesis, label in zip(current_premises, current_hypotheses, current_labels):
            example = prompt + premise + '=>'

            if code == 'zh':
                max_len_b = len(premise) * 1.2 + 100
            else:
                max_len_b = len(premise.split()) * 1.2 + 100

            premise_examples.append(example)
            min_len_b_premises.append(max_len_b)

            example = prompt + hypothesis + ' => '

            if code == 'zh':
                max_len_b = len(hypothesis) * 1.2 + 100
            else:
                max_len_b = len(hypothesis.split()) * 1.2 + 100

            hypotheses_examples.append(example)
            min_len_b_hypotheses.append(max_len_b)

        
        min_len_b = np.max(min_len_b_premises)
        pred_premises = lm.translate(premise_examples, beam=1, max_len_a=1.0,  max_len_b=max_len_b, replace_newlines_with_eos=True)
        min_len_b = np.max(min_len_b_hypotheses)
        pred_hypotheses = lm.translate(hypotheses_examples, beam=1, max_len_a=1.0,  max_len_b=max_len_b, replace_newlines_with_eos=True)
        
        for pred_premise, pred_hypothesis, label in zip(pred_premises, pred_hypotheses, current_labels):
            pred_premise = pred_premise.split('=>')[-1]
            pred_hypothesis = pred_hypothesis.split('=>')[-1]
            f.write('{}\t{}\t{}\t{}\n'.format(code, label, pred_premise, pred_hypothesis))

