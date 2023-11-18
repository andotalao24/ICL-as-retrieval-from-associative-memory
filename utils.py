import json
import numpy as np
import openai
import asyncio
import random
PUNCTUATIONS=['.','?','!',';',':',',','</s>']

def removeNotation(s):
    for p in PUNCTUATIONS:
        s=s.replace(p,'')
    return s.strip()


def requestLLM(**kwargs):
    response = openai.Completion.create(**kwargs)
    ret=[pred['text'] for pred in response['choices']]
    probs=[p['logprobs'] for p in response['choices']]
    #print(response.choices[0].logprobs.token_logprobs)

    #print(response.choices[0])
    return ret,probs


async def requestChat(**kwargs):
    messagelist = kwargs['messages']
    ret=[]
    for message in messagelist:
        kwargs['messages']=message
        ret.append(openai.ChatCompletion.acreate(**kwargs))

    return await asyncio.gather(*ret)


async def requestResponse(modelname,**kwargs):
    if modelname=='chat':
        res = await requestChat(**kwargs)
        return [completion.choices[0].message.content for completion in res],[]

    else:

        return requestLLM(**kwargs)


def writeResult(outputFile,preds,golds,inputs,f1Arr):
    with open(outputFile,'w') as f:
        tmpList=[]
        i=0
        for (pred,gold,input) in zip(preds,golds,inputs):
            tmpList.append({'pred':pred,'gold':gold,'prompt':input})
            i+=1
        print(len(tmpList),len(preds))
        final={'result':tmpList,'acc':np.mean(np.array([i for i in f1Arr if i!=-1]))}
        json.dump(final,f,indent=2)


def appendResult(outputFile,preds,golds,inputs,acc):
    with open(outputFile,'a+') as f:
        tmpList=[]
        for (pred,gold,input) in zip(preds,golds,inputs):
            json.dump({'pred':pred,'gold':gold,'acc':acc},f)
            f.write('\n')
        #final={'result':tmpList,'acc':np.mean(np.array([i for i in f1Arr if i!=-1]))}
        #json.dump(final,f,indent=2)

def rmPunctuation(sent,gold):
    '''
    :param sent: a string,
    :param gold: parsed sent, assume no terminal
    :return: sent, gold with punctuation removed
    '''
    ret_sent=''
    ret_label=''
    punctList=['%',"'",'"','*']
    tokens=[t for t in sent.split() if t not in punctList]

    gold=[t for t in gold.split() if t==')' or t=='(' or t.replace('(','').replace(')','').split()[0] not in punctList]
    for t in tokens:
        ret_sent+=t+' '
    for gt in gold:
        ret_label+=gt+' '
    return ret_sent.strip(), ret_label.strip()


def read_Json(file):

    with open(file,'r', encoding='utf-8') as f:
        return json.load(f)


def read_row(file):
    #return list of dictionaries
    ret=[]
    with open(file,'r', encoding="UTF-8") as f:
        for row in f.readlines():
            d=json.loads(row)
            ret.append(d)
    return ret


def store_row(file,ret):
    with open(file,'w') as f:
        for row in ret:
            json.dump(row,f)
            f.write('\n')


def dict2prompt(d,cot=1,multi=0,index=0,ref_indx=0):
    # format row data into a demonstration
    if cot:
        if multi:
            n=len(d['cot'])
            index = n-1 if index >= n else index
            index = ref_indx if index<0 else index
            exp=d['cot'][index]
        else:
            exp=d['cot']
        prompt='Question: '+d['que']+'\n'+'Explanation: '+exp+'Answer: '+d['ans']+'\n'
    else:
        prompt = 'Question: ' + d['que'] + '\n' + 'Answer: ' + d['ans'] + '\n'
    return prompt


def chat_form(d, cot,multi=0,index=0,ref_indx=0,choice=0):
    if 'que' not in d:
        d=d['data']
    try:
        ans_idx=d['ans'].strip().split(':')[-2].strip()[-1]
        ans=d['ans'].strip().split(':')[-1].strip()
    except:
        ans_idx=''
        ans=d['ans']
    if cot:
        if multi:

            n=len(d['cot'])
            #ref_indx = random.choice(range(n)) #randomly choose a reference optional
            index = n-1 if index >= n else index
            index = ref_indx%n if index<0 else index
            exp=d['cot'][index]
        else:
            exp=d['cot']
        if choice:
            return [{"role": "user", "content":'Question: '+d['que']}, {"role": "assistant", "content": exp + ' The answer is {}: {}'.format(ans_idx,ans)}]
        else:
            return [{"role": "user", "content": 'Question: '+d['que']}, {"role": "assistant", "content": exp+ ' The answer is: {}'.format(ans)}]
    else:
        if choice:
            return [{"role": "user", "content": 'Question: '+d['que']}, {"role": "assistant", "content":' The answer is {}: {}'.format(ans_idx,ans)}]
        else:
            return [{"role": "user", "content": 'Question: '+d['que']}, {"role": "assistant", "content":' The answer is: {}'.format(ans)}]



def getQue_chat(q):
    return {'role': 'user', 'content': q}


def process_medcq(file, make_ans_wrong=0):
    #convert to {question+option; answer; cop}
    # ...question...\n
    # CoT...The answer is \n
    # cop: a/b/c/d
    ops=['A','B','C','D']
    dt=read_row(file)
    print('num of entries. medcq {}'.format(len(dt)))
    ret=[]
    for d in dt:
        if make_ans_wrong:
            d['cop'] = (d['cop'] + 1) % 4

        ans=[d['opa'],d['opb'],d['opc'],d['opd']]
        d['que'] = d['question']+' Choose the answer from A to D. A: {}. B: {}. C: {}. D: {}.'.format(d['opa'],d['opb'],d['opc'],d['opd'])
        d['cot'] = d['exp']
        d['ans'] = 'Among A through D, the answer is {}: {}.'.format(ops[d['cop']-1],ans[d['cop']-1])
        d['gold']={'id':d['cop'],'ans':ans[d['cop']-1]}
        ret.append(d)
    return ret


def store_row(file,ret):
    with open(file,'w') as f:
        for row in ret:
            json.dump(row,f)
            f.write('\n')


def process_usmle(file, make_ans_wrong=1):
    try:
        dt=read_row(file)
    except:
        dt=file
    ret=[]
    print('num of entries. usmle {}'.format(len(dt)))
    for d in dt:
        que=' Choose among A through E. '
        for k,v in d['options'].items():
            que+=k+': '+v+'. '
        d['que'] = d['question'] + que
        d['cot'] = 'Let\'s think step by step.'
        d['ans'] = 'Among A through E, the answer is {}: {}.'.format(d['answer_idx'],d['answer'])
        d['gold']={'id':ord(d['answer_idx'])-ord('A')+1,'ans':d['answer']}
        ret.append(d)
    return ret


def cal_acc(preds,golds,choice=0):
    #pred is supposed to be 'CoT + Among A through D, the answer is {}: {}'
    #gold is a list of dictionary
    n=len(preds)
    N=n
    cor=0
    ansret=[]
    for pred,d in zip(preds,golds):
        #pred is a list due to self-consistency
        idx=0
        if choice:
            try:
                idx=chr(ord('A')+d['gold']['id']-1)
            except:
                idx = chr(ord('A') + d['gold']['gold']['id'] - 1)


        pred=[p.strip().split(':') for p in pred]
        nn=0 #number of non-ans in one pred list
        idx_dict={'A':0,'B':0,'C':0,'D':0,'E':0}
        for p in pred:
            try:
                if choice:
                    pred_idx=p[-2].strip()[-1]
                    if pred_idx in ['a','b','c','d','e']:
                        pred_idx=chr(ord('A')+ord(pred_idx)-ord('a'))
                    idx_dict[pred_idx]+=1
                else:
                    pred_ans=removeNotation(p[-1].strip().lower()) #assume len(pred)==1
            except Exception as e:
                print(e)
                nn+=1
                continue
        pred_idx=max(idx_dict,key=idx_dict.get) #pred idx through voting
        if nn==len(pred):
            n-=1
            print('cannot decide')
            ansret.append(-1)
            continue
        is_pred_correct=0

        if (not choice and pred_ans.strip()==d['ans'].strip()) or (choice and pred_idx==idx):
            cor+=1
            is_pred_correct=1
        if not choice:
            print('pred',pred_ans)
            print('gold',d['ans'])
            print(is_pred_correct)
            print()
        ansret.append(is_pred_correct)
    acc=cor/N
    print('cannot decide: {}'.format(N-n))
    return N-n,acc,ansret

