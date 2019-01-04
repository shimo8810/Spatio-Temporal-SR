import chainer
import chainer.functions as F
from chainer import reporter as reporter_module
from chainer.training import extensions

class SeqVGEvaluator(extensions.Evaluator):
    def evaluate(self):
        target = self._targets['main']

        summary = reporter_module.DictSummary()

        # for name, target in self._targets
