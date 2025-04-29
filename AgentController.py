class AgentController:
    def __init__(self):
        self.skills = {}

    def register_skill(self, skill_name, skill_instance):
        self.skills[skill_name] = skill_instance

    def handle_input(self, skill_name, input_text):
        skill = self.skills.get(skill_name)
        if not skill:
            return "Skill not found."
        return skill.run(input_text)
