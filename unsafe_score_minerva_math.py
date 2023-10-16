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
from concurrent.futures import ProcessPoolExecutor, as_completed


INVALID_ANSWER = "[invalidanswer]"


rexp = re.compile(r'Final Answer: The final answer is(.*?). I hope it is correct.')
def get_unnormalized_answer(text: str):
    text += "I hope it is correct."
    match = rexp.search(text)
    if match:
        return match.group(1).strip()
    else:
        return INVALID_ANSWER


voter = MajorityVotingMixin()
checker = SymbolicMathMixin()


def check_answer(doc, i):
    candidates = doc['metadata']['unprocessed_answers']

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
        votes = None
    else:
        answers = [
            checker.normalize_tex(get_unnormalized_answer(candidate))
            for candidate in candidates
        ]
        
        acc, pass_rate, votes = voter.majority_vote(
            answers,
            correct_answer=doc['answer'],
            is_equiv=checker.is_tex_equiv,
            invalid_answer=INVALID_ANSWER,
        )

        if votes:
            answer = votes[0][0]
        else:
            answer = INVALID_ANSWER

    return (acc, pass_rate, votes, i)


def main(args):
    with open(args.output) as f:
        output = json.load(f)

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

        # we support resumption of scoring from in-progress scoring attempts
        # this loop counts which documents have already been assigned accuracies / scored
        done = 0
        docs_to_process = []
        for i, doc in enumerate(docs[:limit]):
            if 'acc' in doc.keys():
                # then we have already processed this document
                done += 1
                accs.append(doc['acc'])
                pass_rates.append(doc['pass_rate'])
            else:
                docs_to_process.append(i)
        print(f"Results already done for {done} docs")


        with tqdm(total=len(docs[:limit]) - done) as progress:
            # iterate through the remaining unscored documents. 
            # we use multiprocessing to accelerate the process,
            # unless --no_multiprocessing is passed.
            # this may be helpful for debugging, as
            # the tail of possible arcane SymPy errors that may arise
            # when evaluating hundreds of high-temp model generations
            # is quite unexpectedly large.
            if args.no_multiprocessing:
                print(f"Executing last {len(docs_to_process)} documents without multiprocessing")
                for i, doc in enumerate(docs):
                    if i in docs_to_process:
                        res = check_answer(doc, i)
                        accs.append(res[0])
                        pass_rates.append(res[1])

                        output['cache'][task][res[3]]['acc'] = res[0]
                        output['cache'][task][res[3]]['pass_rate'] = res[1]

                        if res[2]: # if is_majority_voting
                            output['cache'][task][res[3]]['votes'] = res[2]

                        # we write results back to our file after each document completes.
                        with open(args.output, 'w') as f:
                            f.write(json.dumps(output, indent=4))

                        progress.update(1)
            else:
                # the typical case: we use multiprocessing to, when possible, speed up answer checking.
                with ProcessPoolExecutor() as executor:
                    futures = [executor.submit(check_answer, doc, i) for i, doc in list(enumerate(docs[:limit])) if i in docs_to_process]
                    for res in as_completed(futures):
                        res = res.result()

                        accs.append(res[0])
                        pass_rates.append(res[1])

                        output['cache'][task][res[3]]['acc'] = res[0]
                        output['cache'][task][res[3]]['pass_rate'] = res[1]

                        if res[2]: # if is_majority_voting
                            output['cache'][task][res[3]]['votes'] = res[2]

                        # we write results back to our file after each document completes.
                        with open(args.output, 'w') as f:
                            f.write(json.dumps(output, indent=4))

                        progress.update(1)

        assert len(accs) == len(output['cache'][task])
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

    parser = argparse.ArgumentParser("Unsafe script for scoring the minerva_math tasks")

    parser.add_argument("--output", type=str, help="path to output file from running minerva_math tasks")
    parser.add_argument("--limit", type=int, default=None, help="for debugging purposes, max examples per task to process")
    parser.add_argument("--no_multiprocessing", action='store_false', help="for debugging, optionally disable multiprocessing.")

    args = parser.parse_args()
    main(args)