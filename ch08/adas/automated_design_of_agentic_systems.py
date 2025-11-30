import argparse
import copy
import json
import os
import pickle
import random
from utils import random_id, bootstrap_confidence_interval
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor


import backoff
import numpy as np
import openai
import pandas
from tqdm import tqdm

# 환경변수
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

client = openai.OpenAI()

# 작업에 맞는 prompt 가져오기
# 새로운 작업을 위해서는 get_init_archive, get_prompt, get_reflexion_prompt 함수를 포함한 새로운 prompt 모듈을 생성하세요.

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.5
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert json_dict is not None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        msg_list,
        model,
        temperature=1
):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=msg_list,
            temperature=temperature, stop=None, response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        
        # 디버깅: 응답 내용 확인
        if PRINT_LLM_DEBUG:
            print(f"[DEBUG] Model: {model}")
            print(f"[DEBUG] Response content: {content[:200] if content else 'EMPTY'}")
            print(f"[DEBUG] Finish reason: {response.choices[0].finish_reason}")
            print(f"[DEBUG] Usage: {response.usage}")
        
        if not content:
            raise ValueError(f"Empty response from model {model}. Finish reason: {response.choices[0].finish_reason}")
        
        json_dict = json.loads(content)
        assert json_dict is not None
        return json_dict
    except Exception as e:
        if PRINT_LLM_DEBUG:
            print(f"[DEBUG] Exception in get_json_response_from_gpt_reflect: {e}")
            print(f"[DEBUG] Messages length: {len(msg_list)}")
        raise


class LLMAgentBase:
    """
    LLM 에이전트의 기본 클래스, 다양한 출력 형식을 위해 설정 가능합니다.
    """
    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model='gpt-5-mini', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id()  # Assume random_id from utils

    def generate_prompt(self, input_infos, instruction, output_description) -> tuple:
        output_fields_and_description = {key: output_description.get(key, f"Your {key}.") for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
                if author == self.__repr__():
                    author += ' (yourself)'
                if field_name == 'task':
                    input_infos_text += f'# Your Task:\n{content}\n\n'
                elif iteration_idx != -1:
                    input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
                else:
                    input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, output_description, iteration_idx=-1) -> list:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction, output_description)
        try:
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            response_json = {key: '' for key in self.output_fields if key not in response_json}
            for key in list(response_json):
                if key not in self.output_fields:
                    del response_json[key]
        output_infos = [Info(key, self.__repr__(), value, iteration_idx) for key, value in response_json.items()]
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, output_description, iteration_idx=-1):
        return self.query(input_infos, instruction, output_description, iteration_idx)


class AgentSystem:
    """
    AgentSystem의 기본 클래스, ARC에서 피드백과 같은 작업별 동작을 확장할 수 있습니다.
    """
    def __init__(self, **task_specific_init):
        for k, v in task_specific_init.items():
            setattr(self, k, v)


class BaseTask:
    """
    작업의 추상 기본 클래스, 새로운 문제를 위해 서브클래스를 생성하세요.
    Required methods:
    - get_init_archive: 초기 솔루션.
    - get_prompt: 새로운 솔루션을 생성하기 위한 프롬프트.
    - get_reflexion_prompt: 반영을 위한 프롬프트.
    - load_data: 검증/테스트 데이터 로드.
    - format_task: 데이터를 프롬프트 문자열로 포맷.
    - get_ground_truth: 데이터에서 진실 추출.
    - evaluate_prediction: 예측과 진실 간의 점수 매기기 (예: 1/0 정확도).
    - parse_prediction: 원시 전방 출력을 비교 가능한 형식으로 파싱.
    - get_output_description: 프롬프트의 출력 필드를 딕셔너리로 가져오기.
    - get_instruction: 프롬프트에 추가 지시사항.

    Optional:
    - prepare_task_queue: 병렬 평가를 위한 입력 준비 (기본: 간단한 리스트).
    - get_agent_system: 사용자 정의 AgentSystem 인스턴스 (기본: 기본).
    """
    def __init__(self, args):
        self.args = args

    def get_init_archive(self):
        raise NotImplementedError

    def get_prompt(self, archive):
        raise NotImplementedError

    def get_reflexion_prompt(self, prev_solution):
        raise NotImplementedError

    def load_data(self, mode):  # mode: True for search (val), False for eval (test)
        raise NotImplementedError

    def format_task(self, task_data):
        raise NotImplementedError

    def get_ground_truth(self, task_data):
        raise NotImplementedError

    def evaluate_prediction(self, prediction, ground_truth):
        raise NotImplementedError

    def parse_prediction(self, res):
        raise NotImplementedError

    def get_output_description(self):
        return {}

    def get_instruction(self):
        return ""

    def prepare_task_queue(self, data):
        return [Info('task', 'User', self.format_task(d), -1) for d in data]

    def get_agent_system(self, task_data=None):
        return AgentSystem()


# MMLU 예제 서브클래스 (이전 코드 기반)
class MMLUTask(BaseTask):
    def get_init_archive(self):
        from mmlu_prompt import get_init_archive  # Task-specific
        return get_init_archive()

    def get_prompt(self, archive):
        from mmlu_prompt import get_prompt
        return get_prompt(archive)

    def get_reflexion_prompt(self, prev_solution):
        from mmlu_prompt import get_reflexion_prompt
        return get_reflexion_prompt(prev_solution)

    def load_data(self, mode):
        df = pandas.read_csv(self.args.data_filename)
        random.seed(self.args.shuffle_seed)
        examples = [row.to_dict() for _, row in df.iterrows()]
        random.shuffle(examples)
        if mode:
            return examples[:self.args.valid_size] * self.args.n_repeat
        else:
            start = self.args.valid_size
            end = start + self.args.test_size
            return examples[start:end] * self.args.n_repeat

    def format_task(self, task_data):
        from utils import format_multichoice_question  # Assume in utils
        return format_multichoice_question(task_data)

    def get_ground_truth(self, task_data):
        LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        return LETTER_TO_INDEX[task_data['Answer']]

    def evaluate_prediction(self, prediction, ground_truth):
        return 1 if prediction == ground_truth else 0

    def parse_prediction(self, res):
        LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        try:
            if isinstance(res, str) and res in LETTER_TO_INDEX:
                return LETTER_TO_INDEX[res]
            if 'A)' in str(res):
                return 0
            if 'B)' in str(res):
                return 1
            if 'C)' in str(res):
                return 2
            if 'D)' in str(res):
                return 3
            if isinstance(res, list) and len(res) > 1:
                content = res[1].content
                return LETTER_TO_INDEX[content] if content in LETTER_TO_INDEX else -1
            if hasattr(res, 'content'):
                content = res.content
                if content in LETTER_TO_INDEX:
                    return LETTER_TO_INDEX[content]
                if 'A)' in content:
                    return 0
                if 'B)' in content:
                    return 1
                if 'C)' in content:
                    return 2
                if 'D)' in content:
                    return 3
        except:
            pass
        return -1

    def get_output_description(self):
        return {'answer': '당신의 답변. 알파벳 선택만 반환하세요, 즉 A 또는 B 또는 C 또는 D.'}


# ARC 예제 서브클래스 (이전 코드 기반)
class ARCTask(BaseTask):
    def get_init_archive(self):
        from arc_prompt import get_init_archive
        return get_init_archive()

    def get_prompt(self, archive):
        from arc_prompt import get_prompt
        return get_prompt(archive)

    def get_reflexion_prompt(self, prev_solution):
        from arc_prompt import get_reflexion_prompt
        return get_reflexion_prompt(prev_solution)

    def load_data(self, mode):
        path = self.args.val_data_path if mode else self.args.test_data_path
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data * self.args.n_repeat

    def format_task(self, task_data):
        from utils import format_arc_data
        task_str, _, _ = format_arc_data(task_data)
        return task_str

    def get_ground_truth(self, task_data):
        return task_data['test'][0]['output']  # ARC 구조

    def evaluate_prediction(self, prediction, ground_truth):
        from utils import eval_solution
        arc_data = {'test': [{'output': ground_truth}]}
        return eval_solution(prediction, arc_data, soft_eval=False)

    def parse_prediction(self, res):
        try:
            if isinstance(res, Info):
                res = res.content
            if isinstance(res, str):
                res = eval(res)
            return res
        except:
            return None

    def get_output_description(self):
        return {'answer': '당신의 답변. list[list[int]] 형식의 문자열만 반환하세요. 다른 것은 반환하지 마세요.'}

    def get_instruction(self):
        return "`transform` 함수를 생성하여 이 작업을 해결하세요. 이 함수는 단일 인수를 받아들이며, 입력 그리드를 `list[list[int]]` 형식으로 받아들이고, 변환된 그리드를 `list[list[int]]` 형식으로 반환해야 합니다."

    def prepare_task_queue(self, data):
        from utils import format_arc_data
        queue = []
        for arc_data in data:
            task_str, examples, test_input = format_arc_data(arc_data)
            taskInfo = Info('task', 'User', task_str, -1)
            agent = self.get_agent_system(examples=examples, test_input=test_input)
            queue.append((agent, taskInfo, arc_data))
        return queue

    def get_agent_system(self, **kwargs):
        from utils import list_to_string  # Assume in utils
        class ARCAgentSystem(AgentSystem):
            def __init__(self, examples, test_input):
                super().__init__(examples=examples, test_input=test_input)

            # run_examples_and_get_feedback 및 get_test_output_from_code가 필요한 경우 추가
            # (omitted for brevity, but can copy from original ARC code)

        return ARCAgentSystem(kwargs['examples'], kwargs['test_input'])


def search(args, task):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            archive = json.load(f)
        last_gen = archive[-1].get('generation', 0) if archive else 0
        start = int(last_gen) if isinstance(last_gen, int) else 0
    else:
        archive = task.get_init_archive()
        start = 0

    # 초기 아카이브 평가
    for solution in archive:
        if 'fitness' in solution:
            continue
        solution['generation'] = "initial"
        print(f"============Initial Archive: {solution.get('name', 'unnamed')}=================")
        try:
            acc_list = evaluate_forward_fn(args, solution["code"], task)
        except Exception as e:
            print(f"Error evaluating initial: {e}")
            continue
        solution['fitness'] = bootstrap_confidence_interval(acc_list)  
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(archive, f, indent=4)

    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        system_prompt, prompt = task.get_prompt(archive)
        msg_list = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        try:
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            ref1, ref2 = task.get_reflexion_prompt(archive[-1] if n > 0 else None)
            msg_list += [{"role": "assistant", "content": str(next_solution)}, {"role": "user", "content": ref1}]
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            msg_list += [{"role": "assistant", "content": str(next_solution)}, {"role": "user", "content": ref2}]
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
        except Exception as e:
            print(f"Error generating solution: {e}")
            continue

        acc_list = []
        for _ in range(args.debug_max):
            try:
                acc_list = evaluate_forward_fn(args, next_solution["code"], task)
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print(f"Evaluation error: {e}")
                msg_list += [{"role": "assistant", "content": str(next_solution)}, 
                             {"role": "user", "content": f"Error: {e}\nDebug and repeat thought in 'thought', debug in 'debug_thought'"}]
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
                except Exception as ee:
                    print(f"Error debugging: {ee}")
                    break
        if not acc_list:
            continue

        next_solution['fitness'] = bootstrap_confidence_interval(acc_list)
        next_solution['generation'] = n + 1
        next_solution.pop('debug_thought', None)
        next_solution.pop('reflection', None)
        archive.append(next_solution)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(archive, f, indent=4)


def evaluate(args, task):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = file_path.replace(".json", "_evaluate.json")
    with open(file_path, 'r') as f:
        archive = json.load(f)
    eval_archive = json.load(open(eval_file_path)) if os.path.exists(eval_file_path) else []

    current_idx = len(eval_archive)
    while current_idx < len(archive):
        sol = archive[current_idx]
        print(f"Evaluating gen {sol['generation']}, idx {current_idx}")
        try:
            acc_list = evaluate_forward_fn(args, sol["code"], task)
            sol['test_fitness'] = bootstrap_confidence_interval(acc_list)
            eval_archive.append(sol)
            with open(eval_file_path, 'w') as f:
                json.dump(eval_archive, f, indent=4)
        except Exception as e:
            print(f"Error: {e}")
        current_idx += 1


def evaluate_forward_fn(args, forward_str, task):
    namespace = {}
    exec(forward_str, globals(), namespace)
    func = list(namespace.values())[0]
    if not callable(func):
        raise ValueError("Not callable")
    AgentSystem.forward = staticmethod(func)  # 클래스에 첨부

    data = task.load_data(SEARCHING_MODE)
    task_queue = task.prepare_task_queue(data)
    max_workers = min(len(task_queue), args.max_workers) if args.multiprocessing else 1

    def process_item(item):
        # ARC task는 (agent, taskInfo, task_data) tuple을 반환하고
        # MMLU task는 Info 객체만 반환합니다.
        # Info는 namedtuple이므로 isinstance(item, tuple)이 True지만
        # 길이가 3이 아니라 4입니다.
        if isinstance(item, tuple) and not isinstance(item, Info):
            agent, taskInfo, task_data = item
        else:
            agent = task.get_agent_system()
            taskInfo = item
            # 해당 데이터 찾기 (인덱스 필요 시)
            idx = task_queue.index(item)
            task_data = data[idx]
        res = agent.forward(taskInfo)
        prediction = task.parse_prediction(res)
        ground_truth = task.get_ground_truth(task_data)
        if prediction is None:
            return 0
        return task.evaluate_prediction(prediction, ground_truth)

    with ThreadPoolExecutor(max_workers) as executor:
        acc_list = list(tqdm(executor.map(process_item, task_queue), total=len(task_queue)))

    print(f"acc: {bootstrap_confidence_interval(acc_list)}")  
    return acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic ADAS Framework")
    parser.add_argument('--task', type=str, required=True, choices=['mmlu', 'arc'], help="Task type")
    # Common
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--expr_name', type=str, default='adas')
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model', type=str, default='gpt-5-mini')
    parser.add_argument('--n_repeat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true')
    parser.add_argument('--max_workers', type=int, default=48)
    # MMLU-specific
    parser.add_argument('--data_filename', type=str)
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    # ARC-specific
    parser.add_argument('--val_data_path', type=str)
    parser.add_argument('--test_data_path', type=str)

    args = parser.parse_args()

    if args.task == 'mmlu':
        task = MMLUTask(args)
        args.expr_name += '_mmlu'
    elif args.task == 'arc':
        if not args.val_data_path or not args.test_data_path:
            print("오류: ARC 태스크는 --val_data_path와 --test_data_path 인자가 필요합니다.")
            exit(1)
        task = ARCTask(args)
        args.expr_name += '_arc'
    else:
        print("Error: Invalid task type. Please choose 'mmlu' or 'arc'.")
        exit(1)

    SEARCHING_MODE = True
    search(args, task)