from . common import NLP_TASK


class WikiText103(NLP_TASK):
    NLP_PATH = "wikitext"
    NLP_NAME = "wikitext-103-raw-v1"

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def doc_to_text(self, doc):
        # TODO: implement
        pass

    def doc_to_target(self, doc):
        # TODO: implement
        pass

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of 
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural 
            language description, as well as the few shot examples, and the question
            part of the document for `doc`. 
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')
    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metrics
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')


class WikiText2(NLP_TASK):
    NLP_PATH = "wikitext"
    NLP_NAME = "wikitext-2-raw-v1"

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def doc_to_text(self, doc):
        # TODO: implement
        pass

    def doc_to_target(self, doc):
        # TODO: implement
        pass

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of 
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural 
            language description, as well as the few shot examples, and the question
            part of the document for `doc`. 
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')
    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metrics
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')
