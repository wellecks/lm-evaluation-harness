"""
MiniF2F in Isabelle

MiniF2F [Zheng et al 2022] is a formal mathematics benchmark (translated across
multiple formal systems) consisting of exercise statements from olympiads
(AMC, AIME, IMO) as well as high-school and undergraduate maths classes.

This version contains formal statements in Isabelle, each paired with an informal
statement and an informal proof as described in Draft, Sketch, Prove [Jiang et al 2023].

The evaluation harness supports the following tasks:
- `minif2f_isabelle`: given a formal statement, generate a formal proof checked by Isabelle.
- `minif2f_isabelle_informal2formal`: given a formal statement, informal statement, informal proof,
   generate a formal proof checked by Isabelle.

The generated formal proof can have `sledgehammer` calls as in Draft, Sketch, Prove.
Proof checking (and calls to sledgehammer or similar automated provers) is handled
via Portal-to-Isabelle [Jiang et al 2021].

Homepage: https://huggingface.co/datasets/wellecks/minif2f_isabelle
"""
from lm_eval.metrics import mean
from lm_eval.tasks.math_tasks import SymbolicMathTask

import os
import sys
import time

_CITATION = """@inproceedings{jiang2023draft,
    title={Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs},
    author={Albert Qiaochu Jiang and Sean Welleck and Jin Peng Zhou and Timothee Lacroix and Jiacheng Liu and Wenda Li and Mateja Jamnik and Guillaume Lample and Yuhuai Wu},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=SMa9EAovKMC}
}

@inproceedings{zheng2022miniff,
    title={miniF2F: a cross-system benchmark for formal Olympiad-level mathematics},
    author={Kunhao Zheng and Jesse Michael Han and Stanislas Polu},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=9ZPegFuFTFv}
}

@article{jiang2021lisa,
  title={LISA: Language models of ISAbelle proofs},
  author={Jiang, Albert Q. and Li, Wenda and Han, Jesse Michael and Wu, Yuhuai},
  year={2021},
  journal={6th Conference on Artificial Intelligence and Theorem Proving},
}
"""

FORMAL2FORMAL_PROMPT=r"""theorem aime_1983_p9:
  fixes x::real
  assumes "0<x" "x<pi"
  shows "12 \<le> ((9 * (x^2 * (sin x)^2)) + 4) / (x * sin x)"
proof -
  define y where "y=x * sin x"
  have "12 \<le> (9 * y^2 + 4) / y"
  proof -
    have c0: "y > 0"
      sledgehammer
    have "(9 * y^2 + 4) \<ge> 12 * y" 
      sledgehammer
    then show ?thesis
      sledgehammer
  qed
  then show ?thesis
    sledgehammer
qed



theorem algebra_3rootspoly_amdtamctambeqnasqmbpctapcbtdpasqmbpctapcbta:
  fixes a b c d :: complex
  shows "(a-d) * (a-c) * (a-b) = -(((a^2 - (b+c) * a) + c * b) * d) + (a^2 - (b+c) * a + c * b) * a"
proof -
  have t0: "a^2 = a * a"
    using power2_eq_square
      sledgehammer
  show ?thesis unfolding t0
    sledgehammer
qed



theorem amc12a_2013_p7:
  fixes s :: "nat \<Rightarrow> real"
  assumes h0 : "\<And>n. s (n+2) = s (n+1) + s n"
    and h1 : "s 9 = 110"
    and h2 : "s 7 = 42"
  shows "s 4 = 10"
proof -
  have "s 8 = 68" using h1 h2 h0[of 7] sledgehammer
  hence h3: "s 6 = 26" using h2 h0[of 6] sledgehammer
  hence "s 5 = 16" using h2 h0[of 5] sledgehammer
  then show ?thesis using h3 h0[of 4] sledgehammer
qed



theorem mathd_numbertheory_188:
  "gcd 180 168 = (12::nat)"
  sledgehammer



theorem numbertheory_2dvd4expn:
  fixes n :: nat
  assumes h0 : "n \<noteq> 0"
  shows "(2::nat) dvd 4^n"
proof -
  obtain m::nat where c0: "m+1=n"
    sledgehammer
  have "(2::nat) dvd 4^(m+1)" sledgehammer
  then show ?thesis unfolding c0 sledgehammer
qed



theorem mathd_numbertheory_412:
  fixes x y :: nat
  assumes h0 : "x mod 19 = (4:: nat)"
    and h1 : "y mod 19 = (7:: nat)"
  shows "(x+1)^2 * (y+5)^3 mod 19 = (13:: nat)"
proof -
  have "(x+1) mod 19 = 5" using h0
    sledgehammer
  then have "(x+1)^2 mod 19 = 5^2 mod 19"
    sledgehammer
  also have "... = 6" sledgehammer
  finally have c0: "(x+1)^2 mod 19 = 6" sledgehammer
  have "(y+5) mod 19 = (7+5) mod 19" using h1
    sledgehammer
  also have "... = 12" sledgehammer
  finally have "(y+5) mod 19 = 12" sledgehammer
  then have "(y+5)^3 mod 19 = 12^3 mod 19"
    sledgehammer
  also have "... = 18" sledgehammer
  finally have c1: "(y+5)^3 mod 19 = 18" sledgehammer
  have "(x+1)^2 * (y+5)^3 mod 19 = ((x+1)^2 mod 19) * ((y+5)^3 mod 19) mod 19"
    sledgehammer
  also have "... = 6 * 18 mod 19" unfolding c0 c1 sledgehammer
  also have "... = 13" sledgehammer
  finally show ?thesis sledgehammer
qed



theorem amc12b_2002_p3:
  fixes n ::nat
  assumes "n>0"
    and prime:"prime (n^2+2-3*n)"
  shows "n=3"
proof -
  have "n>2" 
  proof (rule ccontr)
    assume "\<not> 2 < n"
    then have "n=1 \<or> n=2" using \<open>n>0\<close> sledgehammer
    then show False using prime[THEN prime_gt_1_nat]
      sledgehammer
  qed
  then have "n^2+2-3*n  = (n-1) * (n-2)"
    unfolding power2_eq_square
    sledgehammer
  then have "prime ((n-1) * (n-2))"
    using prime sledgehammer
  then have "n-1=1 \<or> n-2 = 1"
    using prime_product sledgehammer
  with \<open>n>2\<close>
  show "n=3" sledgehammer
qed



theorem mathd_numbertheory_466:
  "(\<Sum> k< 11. k) mod 9 = (1::nat)"
  sledgehammer



theorem mathd_numbertheory_48:
  fixes b :: real
  assumes h0 : "0<b"
    and h1 : "3 * b^2 + 2 * b + 1 = 57"
  shows "b=4"
proof -
  have "0 = 3 * b^2 + 2 * b -56" using h1 sledgehammer
  also have "... = (3*b+14)*(b-4)" sledgehammer
  finally have "0 = (3*b+14)*(b-4)" sledgehammer
  then have "b = -14/3 ∨ b=4" sledgehammer
  then show ?thesis using h0 sledgehammer
qed
end



theorem mathd_numbertheory_335:
  fixes n :: nat
  assumes h0 : "n mod 7 = 5"
  shows "(5 * n) mod 7 = 4"
proof -
  have c0:"(5 * n) mod 7 = (5 * 5) mod 7" using h0
    sledgehammer
  then have "\<dots> = 4" sledgehammer
  then have "(5 * n) mod 7 = 4" using c0 sledgehammer
  then show ?thesis sledgehammer
qed



theorem mathd_numbertheory_284:
  fixes a b :: nat
  assumes h0 : "1\<le>a \<and> a \<le>9 \<and> b \<le>9"
    and h1 : "10 * a + b = 2 * (a+b)"
  shows "10 * a + b = 18"
proof -
  have c0: "8 * a = b" using h1 sledgehammer
  hence "b \<ge> 8" using h0 sledgehammer
  hence c1:"b = 8" using h0 c0
    sledgehammer
  hence "a = 1" using c0 sledgehammer
  then show ?thesis using c1 sledgehammer
qed



theorem algebra_binomnegdiscrineq_10alt28asqp1:
  fixes a :: real
  shows "10 * a \<le> 28 * a^2 + 1"
proof -
  have c0: "0 \<le> 28*a^2 - 10*a + 1"
  proof -
    have c1: "(a - (5/28))^2 = a^2 - 10/28*a + (5/28)^2"
      sledgehammer
    then have c2: "0 \<le> a^2 - 10/28*a + (5/28)^2" using c1
      sledgehammer
    then have c3: "0 \<le> 28*a^2 - 10*a + 28*((5/28)^2)" using c2
      sledgehammer
    then have c4: "0 \<le> 28*a^2 - 10*a + 28*((5/28)*(5/28))" using c3
      sledgehammer
    then have c5: "0 \<le> 28*a^2 - 10*a + (25/28)" using c4
      sledgehammer
    then show ?thesis using c5
      sledgehammer
  qed
  then show ?thesis
    sledgehammer
qed



theorem algebra_2rootsintpoly_am10tap11eqasqpam110:
  fixes a :: complex
  shows "(a-10) * (a+11) = a^2 + a -110"
proof -
  have "(a-10) * (a+11) = a^2 - 10*a + 11*a - 10 *11"
    sledgehammer
  also have "\<dots> = a^2 + a - 10 * 11"
    sledgehammer
  also have "\<dots> = a^2 + a - 110"
    sledgehammer
  finally show ?thesis
    sledgehammer
qed



"""


INFORMAL2FORMAL_PROMPT = """Informal:
(*### Problem

Find the minimum value of $\frac{9x^2\sin^2 x + 4}{x\sin x}$ for $0 < x < \pi$. Show that it is 12.

### Solution

Let $y = x \sin x$. It suffices to show that $12 \leq \frac{9y^2 + 4}{y}.
It is trivial to see that $y > 0$. 
Then one can multiply both sides by $y$ and it suffices to show $12y \leq 9y^2 + 4$.
This can be done by the sum of squares method.*)

Formal:
theorem aime_1983_p9:
  fixes x::real
  assumes "0<x" "x<pi"
  shows "12 \<le> ((9 * (x^2 * (sin x)^2)) + 4) / (x * sin x)"
proof -
  (* Let $y = x \sin x$. *)
  define y where "y=x * sin x"
  (* It suffices to show that $12 \leq \frac{9y^2 + 4}{y}. *)
  have "12 \<le> (9 * y^2 + 4) / y"
  proof -
    (* It is trivial to see that $y > 0$. *)
    have c0: "y > 0"
      sledgehammer
    (* Then one can multiply both sides by $y$ and it suffices to show $12y \leq 9y^2 + 4$. *)
    have "(9 * y^2 + 4) \<ge> 12 * y" 
      sledgehammer
    then show ?thesis
      sledgehammer
  qed
  then show ?thesis
    sledgehammer
qed



Informal:
(*### Problem

Show that for any four complex numbers a, b, c, and d, $(a-d)(a-c)(a-b) = -(((a^2 - a(b+c)) + bc) * d) + (a^2 - a(b+c) + bc) * a$.

### Solution

We first see that $a^2 = a * a$ trivially.
Unfolding this, the main equation holds true when terms are rearranged.*)

Formal:
theorem algebra_3rootspoly_amdtamctambeqnasqmbpctapcbtdpasqmbpctapcbta:
  fixes a b c d :: complex
  shows "(a-d) * (a-c) * (a-b) = -(((a^2 - (b+c) * a) + c * b) * d) + (a^2 - (b+c) * a + c * b) * a"
proof -
  (* We first see that $a^2 = a * a$ trivially. *)
  have t0: "a^2 = a * a"
    using power2_eq_square
      sledgehammer
  (* Unfolding this, the main equation holds true when terms are rearranged. *)
  show ?thesis unfolding t0
    sledgehammer
qed



Informal:
(*### Problem

Find the greatest common factor of 180 and 168. Show that it is 12.

### Solution

This is true by simple evaluation.*)

Formal:
theorem mathd_numbertheory_188:
  "gcd 180 168 = (12::nat)"
  sledgehammer



Informal:
(*### Problem

Show that for positive integer n, 2 divides $4^n$.

### Solution

Since n is positive, we can find a natural number m where $m+1=n$.
Then we can show that 2 divides $4^{m+1}$. The conclusion thus follows.*)

Formal:
theorem numbertheory_2dvd4expn:
  fixes n :: nat
  assumes h0 : "n \<noteq> 0"
  shows "(2::nat) dvd 4^n"
proof -
  obtain m::nat where c0: "m+1=n"
    sledgehammer
  have "(2::nat) dvd 4^(m+1)" sledgehammer
  then show ?thesis unfolding c0 sledgehammer
qed



Informal:
(*### Problem

What is the remainder when $1 + 2 + 3 + 4 + \dots + 9 + 10$ is divided by 9? Show that it is 1.

### Solution

This is true by simple evaluation.*)

Formal:
theorem mathd_numbertheory_466:
  "(\<Sum> k< 11. k) mod 9 = (1::nat)"
  sledgehammer



Informal:
(*### Problem

If $321_{b}$ is equal to the base 10 integer 57, find $b$ given that $b>0$. Show that it is 4.

### Solution

Converting $321_{b}$ to base 10 and setting it equal to 57, we find that  \begin{align*} 3(b^2)+2(b^1)+1(b^0)&=57
\\ 3b^2+2b+1&=57
\\\Rightarrow\qquad 3b^2+2b-56&=0
\\\Rightarrow\qquad (3b+14)(b-4)&=0
\end{align*}This tells us that $b$ is either $-\frac{14}{3}$ or $4$. We know that $b>0$, so $b=4$.*)

Formal:
theorem mathd_numbertheory_48:
  fixes b :: real
  assumes h0 : "0<b"
    and h1 : "3 * b^2 + 2 * b + 1 = 57"
  shows "b=4"
proof -
  (* Converting $321_{b}$ to base 10 and setting it equal to 57, we find that  \begin{align*} 3(b^2)+2(b^1)+1(b^0)&=57
  \\ 3b^2+2b+1&=57
  \\\Rightarrow\qquad 3b^2+2b-56&=0
  \\\Rightarrow\qquad (3b+14)(b-4)&=0
  \end{align*} *)
  have "0 = 3 * b^2 + 2 * b -56" using h1 sledgehammer
  also have "... = (3*b+14)*(b-4)" sledgehammer
  finally have "0 = (3*b+14)*(b-4)" sledgehammer
  (* This tells us that $b$ is either $-\frac{14}{3}$ or $4$. *)
  then have "b = -14/3 ∨ b=4" sledgehammer
  (* We know that $b>0$, so $b=4$. *)
  then show ?thesis using h0 sledgehammer
qed

end



Informal:
(*### Problem

When Rachel divides her favorite number by 7, she gets a remainder of 5. What will the remainder be if she multiplies her favorite number by 5 and then divides by 7? Show that it is 4.

### Solution

Let $n$ be Rachel's favorite number. 
Then $n \equiv 5 \pmod{7}$, so $5n \equiv 5 \cdot 5 \equiv 25 \equiv 4 \pmod{7}$.
*)

Formal:
theorem mathd_numbertheory_335:
  fixes n :: nat
  assumes h0 : "n mod 7 = 5"
  shows "(5 * n) mod 7 = 4"
proof -
  (* Then $n \equiv 5 \pmod{7}$, so $5n \equiv 5 \cdot 5 \equiv 25 \equiv 4 \pmod{7}$. *)
  have c0:"(5 * n) mod 7 = (5 * 5) mod 7" using h0
    sledgehammer
  then have "\<dots> = 4" sledgehammer
  then have "(5 * n) mod 7 = 4" using c0 sledgehammer
  then show ?thesis sledgehammer
qed



Informal:
(*### Problem

What positive two-digit integer is exactly twice the sum of its digits? Show that it is 18.

### Solution

We simplify $10a + b = 2(a+b)$ to get $8a = b$.
Since $a$ is at least 1, $b$ is at least 8.
We know $b$ is 8 since $8a = b$ and $a$ is a natural number.
Hence $a$ is 1.
The two-digit integer is hence $18$.
*)

Formal:
theorem mathd_numbertheory_284:
  fixes a b :: nat
  assumes h0 : "1\<le>a \<and> a \<le>9 \<and> b \<le>9"
    and h1 : "10 * a + b = 2 * (a+b)"
  shows "10 * a + b = 18"
proof -
  (* We simplify $10a + b = 2(a+b)$ to get $8a = b$. *)
  have c0: "8 * a = b" using h1 sledgehammer
  (* Since $a$ is at least 1, $b$ is at least 8. *)
  hence "b \<ge> 8" using h0 sledgehammer
  (* We know $b$ is 8 since $8a = b$ and $a$ is a natural number. *)
  hence c1:"b = 8" using h0 c0
    sledgehammer
  (* Hence $a$ is 1. *)
  hence "a = 1" using c0 sledgehammer
  (* The two-digit integer is hence $18$. *)
  then show ?thesis using c1 sledgehammer
qed



Informal:
(*### Problem

Show that for any complex number a, $(a-10)(a+11) = a^2 + a - 110$.

### Solution

We first expand all terms of the left hand side to get $a^2 - 10a + 11a - 10*11$.
This equals $a^2 + a - 10*11 = a^2 + a - 110$.*)

Formal:
theorem algebra_2rootsintpoly_am10tap11eqasqpam110:
  fixes a :: complex
  shows "(a-10) * (a+11) = a^2 + a -110"
proof -
  (* We first expand all terms of the left hand side to get $a^2 - 10a + 11a - 10*11$. *)
  have "(a-10) * (a+11) = a^2 - 10*a + 11*a - 10 *11"
    sledgehammer
  (* This equals $a^2 + a - 10*11 = a^2 + a - 110$. *)
  also have "\<dots> = a^2 + a - 10 * 11"
    sledgehammer
  also have "\<dots> = a^2 + a - 110"
    sledgehammer
  finally show ?thesis
    sledgehammer
qed



"""


class MiniF2FIsabelle(SymbolicMathTask):
    VERSION = 0
    DATASET_PATH = "wellecks/minif2f_isabelle"

    IN_KEY = "formal_statement"

    @property
    def end_seq(self) -> str:
        return "\n\n\n"

    def has_training_docs(self):
        return False

    def get_unnormalized_answer(self, text: str) -> str:
        """
        Arguments:
            text (str): model sample
        Returns:
            out (str | Literal[self.INVALID_ANSWER]): string containing a TeX Expression or
                `self.INVALID_ANSWER`.
        """
        return text

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        if self.has_validation_docs():
            return [
                x for x in self.dataset["validation"]
                if ('theorem ' + x['problem_name'] + ':') not in FORMAL2FORMAL_PROMPT
            ]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return doc[self.IN_KEY]

    def training_docs(self):
        pass

    def doc_to_target(self, doc):
        # no ground-truth targets available in this task
        target = ""
        return target

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        ctx = FORMAL2FORMAL_PROMPT + self.doc_to_text(doc)
        return ctx

    # def process_results(self, doc, results, params={}):
    #     proof = self._parse_result(results[0])
    #     checking_result = self._check_proof(doc, proof, params)
    #     results = {
    #         "success": float(checking_result['success']),
    #         "metadata": {
    #             'proof': proof,
    #             'statement_and_proof': doc['formal_statement'] + proof,
    #             'checking_result': checking_result
    #         }
    #     }
    #     return results

    def process_results(self, doc, results, params={}):
        candidates = results[0]

        assert isinstance(params, dict)
        proofs = []
        checking_results = []
        if self.MAJORITY_VOTING not in params:
            proof = self._parse_result(candidates)
            checking_result = self._check_proof(doc, proof, params)

            if checking_result['success']:
                acc = 1
            else:
                acc = 0
            pass_rate = acc
            proofs.append(proof)
            checking_results.append(checking_result)
        else:
            checking_results = []
            import ipdb; ipdb.set_trace(context=20)
            for candidate in candidates:
                proof = self._parse_result(candidate)
                checking_result = self._check_proof(doc, proof, params)
                proofs.append(proof)
                checking_results.append(checking_result)

            answers = [1.0 if c['success'] else 0.0 for c in checking_results]

            acc, pass_rate, votes = self.majority_vote(
                answers,
                correct_answer=1.0
            )

        results = {
            "acc": acc,
            "pass_rate": pass_rate,
            "metadata": {
                'statement': doc['formal_statement'],
                'proofs': proofs,
                'checking_results': checking_results
            }
        }

        return results

    def aggregation(self):
        return {"acc": mean, "pass_rate": mean}

    def higher_is_better(self):
        return {"acc": True, "pass_rate": True}

    def _check_proof(self, doc, proof, params):
        # Check the proof
        self._validate_params(params)
        settings = params['isabelle_checker']
        checker = Checker(
            working_dir=settings['working_dir'],
            isa_path=settings['isa_path'],
            theory_file=settings['theory_file'],
            port=settings['port']
        )
        formal_statement = doc['formal_statement']
        theorem_with_proof = f"{formal_statement}\n{proof}"
        result = checker.check(theorem_with_proof)
        if result['success']:
            print("==== SUCCESS!!")
            print(theorem_with_proof)
        return result

    def _validate_params(self, params):
        if ('isabelle_checker' not in params or
             any([field not in params['isabelle_checker']
                  for field in ['working_dir', 'isa_path', 'theory_file', 'port']])):
            raise ValueError(
                'The "isabelle_checker" config field needs to be specified; '
                'see docs/isabelle_setup.md for instructions.'
            )

    def _parse_result(self, result):
        return result


class MiniF2FIsabelleInformal2Formal(MiniF2FIsabelle):
    VERSION = 0
    DATASET_PATH = "wellecks/minif2f_isabelle"
    STOP = "\n\n\n"

    def validation_docs(self):
        if self.has_validation_docs():
            return [
                x for x in self.dataset["validation"]
                if ('theorem ' + x['problem_name'] + ':') not in INFORMAL2FORMAL_PROMPT
            ]

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        ctx = (INFORMAL2FORMAL_PROMPT +
               'Informal:\n(*### Problem\n\n' + doc['informal_statement'] + '\n\n' +
               '### Solution\n\n' + doc['informal_proof'] + ' *)' +
               '\n\nFormal:\n' + doc['formal_statement'])
        return ctx


class Checker(object):
    """A modified version of the Draft, Sketch, Prove proof-checking client.
    (https://github.com/albertqjiang/draft_sketch_prove/blob/main/autoformalization/checker.py)

    This checker supports Isabelle2022 via the new version of PISA
    (https://albertqjiang.github.io/Portal-to-ISAbelle/).

    It supports checking a miniF2F-style proof via `check`, or an
    Archive of Formal Proofs (AFP)-style proof via `check_afp`.

    Finally, it replaces `sledgehammer` with a call to `normalhammer`.
    """
    def __init__(self, working_dir, isa_path, theory_file, port=9000):
        sys.path.append(os.environ['PISA_PATH'])
        try:
            from pisa_client import initialise_env
            self.initialise_env = initialise_env
        except:
            print("Set $PISA_PATH to /yourpath/to/Portal-to-ISAbelle/src/main/python")

        self.working_dir = working_dir
        self.isa_path = isa_path
        self.theory_file = theory_file
        self.port = port

    def _initialize(self):
        env = self.initialise_env(
            self.port,
            isa_path=self.isa_path,
            theory_file_path=self.theory_file,
            working_directory=self.working_dir
        )
        return env

    def _exit(self, env):
        try:
            env.post('exit')
        except:
            print("env.post('exit') timed out")
            pass
        os.system("ps aux | grep Isabelle | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")
        os.system("ps aux | grep poly | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")

    def _parse_output(self, obs):
        """Parse the sledgehammer output, otherwise return an empty string"""
        if '<hammer>' in obs:
            output = obs.split('<hammer>')[0]
        else:
            output = ''
        return output

    def _run_step(self, step, i, tls_name, env):
        obs, reward, done, metadata = env.step_to_top_level_state(
            action=step,
            tls_name=tls_name,
            new_name='default_%d' % i
        )
        error = None
        if 'error:' in obs or 'Step error' in obs or 'Unknown error' in obs:
            error = obs
        return obs, reward, done, metadata, error

    def _run_sledgehammer(self, step, i, tls_name, env):
        # First try heuristics
        for heuristic in ['by auto', 'by simp', 'by blast', 'by fastforce', 'by force', 'by eval', 'by presburger', 'by sos', 'by arith', 'by linarith', 'by (auto simp: field_simps)']:
            step_ = step.replace('normalhammer', heuristic)
            obs, reward, done, metadata, error = self._run_step(step_, i, tls_name, env)
            if error is None:
                obs = '%s <hammer> %s' % (heuristic, obs)
                return obs, reward, done, metadata, error
        # Try sledgehammer
        out = self._run_step(step, i, tls_name, env)
        return out

    def check_afp(self, statement, proof):
        # Initialize environment
        env = self._initialize()
        env.proceed_after(statement)
        env.initialise()

        # Parse proof
        steps = Checker.get_parsed(env, proof)

        result = self._check(env, steps)
        return result

    def check(self, statement_and_proof):
        # Initialize environment
        env = self._initialize()
        env.initialise()

        # Wrap and parse theorem
        theory = Checker.wrap_theorem(statement_and_proof)
        steps = Checker.get_parsed(env, theory)

        result = self._check(env, steps)
        return result

    def _check(self, env, steps):
        done = False
        reason = ''
        success = False
        step_results = []
        tls_name = 'default'
        for i, step in enumerate(steps):
            try:
                time0 = time.time()
                if 'normalhammer' in step:
                    obs, reward, done, metadata, error = self._run_sledgehammer(step, i, tls_name, env)
                else:
                    obs, reward, done, metadata, error = self._run_step(step, i, tls_name, env)
                step_time = time.time() - time0
                step_results.append(dict(index=i, step=step, output=self._parse_output(obs), step_time=step_time))
                if error is not None:
                    reason = error
                    success = False
                    done = False
                    break
            except:
                # Timeout - end the proof attempt
                success = False
                done = False
                reason = 'timeout (%d)' % len(step_results)
                step_results.append(dict(index=i, step=step, output=''))
                break

            # Change when successful
            tls_name = 'default_%d' % i

        if done and reward == 1.0:
            success = True

        result = {
            'success': success,
            'reason': reason,
            'num_steps': len(steps),
            'last_step': len(step_results),
            'step_results': step_results
        }
        # Exit environment
        self._exit(env)
        return result

    @staticmethod
    def wrap_theorem(theorem):
        return 'theory Interactive imports HOL.HOL Complex_Main "HOL-Library.Code_Target_Numeral" "HOL-Library.Sum_of_Squares" "Symmetric_Polynomials.Vieta" "HOL-Computational_Algebra.Computational_Algebra" "HOL-Number_Theory.Number_Theory" \n begin\n%s' % theorem

    @staticmethod
    def get_parsed(env, theory, tls_name='default'):
        # HACK: the parsing doesn't work well with `normalhammer`, so we replace
        # all hammer calls with sorry, then replace sorry to normalhammer after parsing.
        theory = theory.replace('sledgehammer', 'sorry')
        theory = theory.replace('normalhammer', 'sorry')

        steps = env.post(f"<parse text> ${theory}")
        steps = steps.split('<SEP>')
        steps = [s for s in steps if s.strip() != '']
        # remove weird '$' step and whitespace steps
        steps = [s for s in steps if s != '$' and s.strip() != '']
        steps = [s.replace('sorry', 'normalhammer') for s in steps]
        return steps



