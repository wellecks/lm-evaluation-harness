import argparse
import logging
import json
import textwrap
from lm_eval.mixins import SymbolicMathMixin, MajorityVotingMixin
from lm_eval.utils import timeout
from lm_eval.evaluator import make_table
from tqdm import tqdm
import copy
from typing import Union
from numpy import isclose, isfinite
import sympy
from functools import partial
import code
import re

INVALID_ANSWER = "[invalidanswer]"

rexp = re.compile(r'Final Answer: The final answer is(.*?). I hope it is correct.')
def get_unnormalized_answer(text: str):
    text += "I hope it is correct."
    match = rexp.search(text)
    if match: 
        return match.group(1).strip()
    else:
        return INVALID_ANSWER

def main(args):
    with open(args.output) as f:
        output = json.load(f)

    voter = MajorityVotingMixin()
    checker = SymbolicMathMixin()

    tasks = [task for task in output['versions'] if "minerva_math" in task]

    results = {}
    for task in tasks:
        logging.info(f"Scoring task {task}")

        docs = output['cache'][task]

        if args.limit:
            limit = args.limit
        else:
            limit = len(docs)

        accs = []
        pass_rates = []
        for i, doc in enumerate(tqdm(docs[:limit])):
            
            candidates = doc['metadata']['candidates']


            is_majority_voting = not isinstance(candidates, str)

            if not is_majority_voting:
                unnormalized_answer = get_unnormalized_answer(candidates)
                answer = checker.normalize_tex(unnormalized_answer)

                if unnormalized_answer==INVALID_ANSWER:
                    acc = 0
                elif checker.is_tex_equiv(answer, doc['answer']):
                    acc = 1 
                else: 
                    acc = 0 

                pass_rate = acc
            else:
                answers = [
                        checker.normalize_tex(get_unnormalized_answer(candidate))
                        for candidate in candidates
                ]
                answers = [candidate for candidate in candidates if candidate!=INVALID_ANSWER]
                     
                acc, pass_rate, votes = voter.majority_vote(
                        answers,
                        correct_answer=doc['answer'],
                        is_equiv=checker.is_tex_equiv,
                )
                if votes:
                    answer = votes[0][0]
                else: 
                    answer = INVALID_ANSWER


            accs.append(acc)
            pass_rates.append(pass_rate)

            output['cache'][task][i]['acc'] = acc
            output['cache'][task][i]['pass_rate'] = pass_rate

            if is_majority_voting: 
                output['cache'][task][i]['votes'] = votes

    
        results[task] = {"acc": sum(accs)/len(accs), "pass_rate": sum(pass_rates)/len(pass_rates)}

    output['results'] = results

    with open(args.output, 'w') as f:
        f.write(json.dumps(output, indent=4))

    print(make_table(output))

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    logging.critical(
            "THIS PROGRAM EXECUTES UNTRUSTED MODEL GENERATED CODE."
            "THERE HAS BEEN NO EFFORT TO AVOID OS AND NETWORK SIDE EFFECTS."
            "USE WITH CAUTION."
    )

    parser = argparse.ArgumentParser("Unsafe script for scoring the sympy_math tasks")

    parser.add_argument("--output", type=str, help="path to output file from running sympy math tasks")
    parser.add_argument("--limit", type=int, default=None, help="for debugging purposes, max examples per task to process")

    args = parser.parse_args()
    main(args)
