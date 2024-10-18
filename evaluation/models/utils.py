import json
import time
import argparse
import re
import pyarrow.parquet as pq
import glob
from tqdm import tqdm
from collections import namedtuple
from PIL import Image

instruction_pool_direct_mc = ['Answer the question with the letter.',
                              'Answer with the letter.',
                              'Answer with the letter of the right choice.',
                              'Respond with the letter of the correct option.',
                              "Choose the letter corresponding to the right answer."
                              ]

instruction_pool_concise_ans = ['Give a concise answer.',
                                "Answer with a single word or phrase.",
                                "Keep it brief.",
                                "Write a very short answer.",
                                "Short answer required.",
                                "Give a very brief answer.",
                                'Be succinct.',
                                "Quick response, please."]



def cal_acc(anses, gts):
    rws = []
    total = len(anses)
    assert len(anses) == len(gts)
    correct, correct_in_domain, correct_out_domain = 0, 0, 0
    for pid in range(total):
        ans = anses[pid]
        gt = gts[pid]
        try:
            ans = float(ans)
        except:
            pass
        try:
            gt = float(gt)
        except:
            pass
        if ans == gt:
            correct += 1
            rws.append(True)
            continue
        else:
            # print(f"pid: {pid} | gt: {gts[pid]} | ans: {anses[pid]}")
            pass
        rws.append(False)

    return correct / total, rws, total


def option_refine(options):
    refined_options = []
    option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for oid, option in enumerate(options):
        option = f"{option_letters[oid]}. {option}"
        refined_options.append(option)
    return refined_options


def extract_one_ans_math(ans):
    ans = ans.strip()
    if ans.endswith('.'):
        ans = ans.strip('.')
    if len(ans) == 1:
        return ans
    try:
        fans = float(ans)
        return fans
    except:
        pass
    if "The answer is " in ans:
        ans_extract = ans.split('The answer is ')[-1].rstrip('.')
        try:
            fans = float(ans_extract)
            return fans
        except:
            pass
        ans = ans_extract
        # ans_extract = ans_extract.replace('(', '').replace(')', '')
        # return ans_extract

    match_ABC = re.search(r'\(([A-G])\)', ans)
    if match_ABC:
        res = match_ABC.group(1)
        res = res[:-1] if res.endswith('.') else res
        return res
    match_isABC = re.search(r'is ([A-G])\.', ans)
    if match_isABC:
        res = (match_isABC.group(1))
        res = res[:-1] if res.endswith('.') else res
        return res

    match_is_num = re.search(r'is (\d+)', ans)
    if match_is_num:
        return match_is_num.group(1)
    match_equal_num = re.search(r'=(.*?)(\d+)', ans)
    if match_equal_num:
        return match_equal_num.group(2)
    match_str_num = re.search(r'\d+\.\d+', ans)
    if match_str_num:
        return match_str_num.group(0)
    match_num = re.search(r'\d+', ans)
    if match_num:
        return match_num.group(0)
    match_ABClast = re.search(r'\b[A-G]\b\.?$', ans)
    if match_ABClast:
        res = match_ABClast.group(0)
        res = res[:-1] if res.endswith('.') else res
        return res

    match_ans_is = re.search(r'\b(\d+)\b\.?$', ans)
    if match_ans_is:
        res = (match_ans_is.group(1))
        res = res[:-1] if res.endswith('.') else res
        return res

    try:
        fans = float(ans.split(' ')[0])
        return fans

    except:
        pass
    return ans


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(path):
    samples = []
    with open(path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def parquet2list(all_files):
    all_data = []
    # all_files = glob.glob("/apdcephfs_us/share_300814644/data/nlp/pretrain_data/MMMU/*/val*.parquet")
    if isinstance(all_files, list):
        for i, f in enumerate(all_files):
            all_data += parquet2list(f)
        return all_data

    table = pq.read_table(all_files)
    df = table.to_pandas()
    list_of_dicts = df.to_dict('records')
    all_data.extend(list_of_dicts)
    return all_data


def check_image_mode(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        print('converting：', img_path)
        img = img.convert('RGB')
        img.save(img_path)
        print('converted：', img_path)

    # return None


def save_binary_image(binary_data, img_name):
    with open(img_name, 'wb') as file:
        file.write(binary_data)


def sort_key(filename):
    # 使用正则表达式提取文件名中的数字部分
    match = re.search(r'_(\d+)\.jpg$', filename)
    # 如果找到匹配的数字部分，则返回这个数字作为排序键
    return int(match.group(1)) if match else float('inf')


def write_jsonl(path, samples, format=False):
    with open(path, 'w') as f:
        for sample in samples:
            if format:
                f.write(json.dumps(sample, indent=4, ensure_ascii=False) + '\n')
            else:
                f.write(json.dumps(sample) + '\n')


def write_json(path, samples, format=False):
    with open(path, 'w') as f:
        if format:
            json.dump(samples, f, indent=4, ensure_ascii=False)
        else:
            json.dump(samples, f)


def format_ans(ans):
    return ans.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')


def remove_duplicates(dict_list):
    seen_ids = set()
    unique_dicts = []
    for item in dict_list:
        if item['id'] not in seen_ids:
            unique_dicts.append(item)
            seen_ids.add(item['id'])

    return unique_dicts


word_num_dic = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
                'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
                'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
                'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
                'ninety': 90}


def ans_match(gemini_short_ans, gt):
    gemini_short_ans = str(gemini_short_ans)
    reformed_ans = extract_one_ans_math(gemini_short_ans)
    try:
        if float(reformed_ans) == float(gt) or f"{float(reformed_ans):.2f}" == f"{float(gt):.2f}":
            return True
    except:
        pass
    reformed_gt = str(gt).replace('(', '').replace(')', '')
    if reformed_ans == gt or gemini_short_ans == gt or reformed_ans == reformed_gt or gemini_short_ans == reformed_gt:
        return True
    if isinstance(reformed_ans, str):
        reformed_ans = reformed_ans.lower()
        for word in word_num_dic.keys():
            if word in reformed_ans:
                return ans_match(word_num_dic[word], reformed_gt)

    # print(f"gemini_short_ans: {gemini_short_ans} gt: {gt}\n"
    #       f"reformed_ans: {reformed_ans} reformed_gt: {reformed_gt}")
    return False


def get_time():
    time_now = time.time()
    return str(time_now).split('.')[0]


def load_mathvista_data():
    ques_data = read_json("./data/testmini.json")
    math_args = {"shot_num": 0, "shot_type": "solution", "use_caption": False, "use_ocr": False}
    MyNamedTuple = namedtuple('MyNamedTuple', math_args.keys())
    my_namedtuple = MyNamedTuple(**math_args)
    all_data_processed = create_query_data(ques_data, {}, {}, my_namedtuple)
    img_folder = '/afs/crc.nd.edu/user/m/mjia2/repos/MathVista/data/images/'

    return ques_data, all_data_processed, img_folder


def load_mathverse_data():
    ques_data = read_json("./mathverse/testmini.json")
    # math_args = {"shot_num": 0, "shot_type": "solution", "use_caption": False, "use_ocr": False}
    # MyNamedTuple = namedtuple('MyNamedTuple', math_args.keys())
    # my_namedtuple = MyNamedTuple(**math_args)
    # all_data_processed = create_query_data(ques_data, {}, {}, my_namedtuple)
    img_folder = './mathverse/images/'
    return ques_data, img_folder


# def load_model_blipflant5(name="Salesforce/blip2-flan-t5-xxl"):
#     processor = Blip2Processor.from_pretrained(name)
#     model = Blip2ForConditionalGeneration.from_pretrained(
#         "Salesforce/blip2-flan-t5-xxl", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
#     )  # doctest: +IGNORE_RESULT
#     return processor, model
def retain_n_images(text, n):
    """
    保留字符串中前 n 个 <image> 标签，移除其余的 <image> 标签。

    参数：
        text (str): 输入的字符串。
        n (int): 需要保留的 <image> 标签数量。

    返回：
        str: 处理后的字符串。
    """
    count = 0

    def replacer(match):
        nonlocal count
        count += 1
        if count <= n:
            return match.group(0)
        else:
            return ''

    # 使用正则表达式查找所有的 <image> 标签
    pattern = r'<image>'
    result = re.sub(pattern, replacer, text)
    return result

def write_res(output_file_path, anses, details):
    with open(output_file_path, 'w') as aw:
        aw.writelines("\n".join(anses))

    output_save_data = output_file_path.replace('.txt', '.json')
    with open(output_save_data, 'w') as aw:
        json.dump(details, aw, indent=4, ensure_ascii=False)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--pid", type=str)
    parser.add_argument("--des", type=str, default='')
    parser.add_argument('-m', "--max_token", type=int, default=256)
    parser.add_argument('-t', "--temperature", type=float, default=0.0)
    parser.add_argument('-e', '--exp', type=str, default="", help='exp name')
    parser.add_argument('-l', '--lora_model_name', type=str, default=None)
    parser.add_argument('-c', '--lora_cpt', type=str)
    parser.add_argument('-s', '--split', type=int)
    parser.add_argument('-p', '--prompt', type=str, default='cot')

    args = parser.parse_args()
    return args
