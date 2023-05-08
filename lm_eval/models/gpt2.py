import transformers
import torch
import math
from lm_eval.base import BaseLM


class HFLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
        ).to(self.device)
        self.gpt2.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
        )

        #assert isinstance(
        #    self.tokenizer,
        #    (
        #        transformers.GPT2Tokenizer,
        #        transformers.GPT2TokenizerFast,
        #        transformers.T5Tokenizer,
        #        transformers.T5TokenizerFast,
        #    ),
        #), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        if isinstance(
            self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)
        ):
            assert self.tokenizer.encode("hello\n\nhello") == [
                31373,
                198,
                198,
                31373,
            ], self.tokenizer.encode("hello\n\nhello")

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.gpt2.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.gpt2(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id, num_return_sequences=1, temperature=0., num_return_sequences_batch=None):
        assert (isinstance(num_return_sequences, int) and num_return_sequences >= 1), f"Incorrect number of candidates to generate: {num_return_sequences}"
        assert temperature >= 0., f"Negative sampling temperature: {temperature}"
        
        # Whether to sample or to decode greedily
        do_sample = (temperature != 0.)
        if not do_sample:
            # If decoding greedily, only sample once
            assert num_return_sequences == 1, f"Decoding greedily but {num_return_sequences} generations"

        if num_return_sequences_batch is not None and num_return_sequences > 1:
            context = context.expand(num_return_sequences_batch, context.shape[1])
            generated_vectors = [
                self.gpt2.generate(
                    context, max_length=max_length, eos_token_id=eos_token_id,
                    do_sample=do_sample, temperature=temperature
                ) for _ in range(math.ceil(num_return_sequences/num_return_sequences_batch))
            ]
            # Pad the generated vectors such that they have the same length
            max_length = max(element.size(1) for element in generated_vectors)
            padded_vectors = []
            for vector in generated_vectors:
                if vector.size(1) < max_length:
                    vector = torch.cat([vector, torch.zeros(vector.size(0), max_length-vector.size(1), dtype=torch.int32, device=self._device)], dim=1)
                padded_vectors.append(vector)
            return torch.cat(padded_vectors, dim=0)[:num_return_sequences]
        
        assert num_return_sequences_batch is None

        if num_return_sequences > 1:
            context = context.expand(num_return_sequences, context.shape[1])
        return self.calculator_generate(
            context, max_length=max_length, eos_token_id=eos_token_id,
            do_sample=do_sample, temperature=temperature
        )

    def calculator_generate(self, context, max_length=None, eos_token_id=None, do_sample=None, temperature=None):

        
        from contextlib import contextmanager
        import signal
        import torch as th

        def use_calculator(sample):
            if "<<" not in sample:
                return None

            parts = sample.split("<<")
            remaining = parts[-1]
            if ">>" in remaining:
                return None
            if "=" not in remaining:
                return None
            lhs = remaining.split("=")[0]
            lhs = lhs.replace(",", "")
            if any([x not in "0123456789*+-/.()" for x in lhs]):
                return None
            try:
                return ast.literal_eval(lhs)
            except:
                return None

        ctx = context
        for _ in range(max_length - context.shape[1]):
            out = self.gpt2.generate(
                ctx, max_length=ctx.shape[1] + 1, eos_token_id=eos_token_id,
                do_sample=do_sample, temperature=temperature, 
            )
            text = self.tokenizer.batch_decode(out)[0]
            if "=" in text[-2:]: # hacky: 
                print(text)
                answer = use_calculator(text)
                if answer is not None:
                    print(f"Triggered calculator on text {text}, answer", answer)
                    text = text + str(answer) + ">>"
                    ctx = self.tokenizer([text], padding=False, return_tensors="pt").to(self._device)
                else: 
                    ctx = out
            else:
                ctx = out

        return ctx
        # def sample(model, qn, tokenizer, device, sample_len):
        #     # Inefficient version of calculator sampling -- no batches, doesn't
        #     # cache activations from previous tokens
        #     EQUALS_TOKENS = set([28, 796, 47505])

        #     for _ in range(sample_len):
        #         with th.no_grad():
        #             toks = tokenizer([qn], padding=False, return_tensors="pt").to(device)
        #             orig_len = toks["input_ids"].shape[1]

        #             out = model.generate(
        #                 **toks, max_length=orig_len + 1, pad_token_id=model.config.eos_token_id
        #             )
        #             text = tokenizer.batch_decode(out)[0]

        #             if out[0, -1].item() in EQUALS_TOKENS:
        #                 answer = use_calculator(text)
        #                 if answer is not None:
        #                     print("Triggered calculator, answer", answer)
        #                     text = text + str(answer) + ">>"

        #             qn = text
        #     return qn


# for backwards compatibility
GPT2LM = HFLM
