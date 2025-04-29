from InputPayload import InputPayload

class BaseSkill:
    def analyze(self, payload: InputPayload):
        raise NotImplementedError

    def decide(self, analysis_result):
        raise NotImplementedError

    def act(self, decision_result):
        raise NotImplementedError

    def run(self, payload: InputPayload):
        analysis = self.analyze(payload)
        decision = self.decide(analysis)
        action = self.act(decision)
        return action
