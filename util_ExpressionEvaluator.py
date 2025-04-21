import ast
import operator
import pandas as pd

class ExpressionCalculator:
    def __init__(self):
        self.memory_result = 0.0
        self.variables = {}
        self.history = []  # Stores (expression, result)

    def set_vars(self, var_dict):
        self.variables = var_dict

    def get_memory(self):
        return self.memory_result

    def evaluate(self, expression):
        try:
            expr_ast = ast.parse(expression, mode='eval')
            result = self._eval_ast(expr_ast.body)
            self.memory_result = result
            self.history.append((expression, result))
            return result
        except Exception as e:
            error_msg = f"Error: {e}"
            self.history.append((expression, error_msg))
            return error_msg

    def _eval_ast(self, node):
        if isinstance(node, ast.BinOp):
            left = self._eval_ast(node.left)
            right = self._eval_ast(node.right)

            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv
            }

            return ops[type(node.op)](left, right)

        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n

        elif isinstance(node, ast.Constant):  # Python 3.8+
            return node.value

        elif isinstance(node, ast.Name):
            if node.id in self.variables:
                return self.variables[node.id]
            else:
                raise ValueError(f"Variable '{node.id}' is not defined")

        else:
            raise TypeError(f"Unsupported AST node: {type(node)}")

    def show_history(self, last_n=None):
        if not self.history:
            return ["No history yet."]

        history_slice = self.history[-last_n:] if last_n else self.history
        return [
            f"{idx+1}: {expr} = {result}"
            for idx, (expr, result) in enumerate(history_slice)
        ]

    def clear_history(self):
        self.history = []
        return "History cleared."

    def export_history_to_csv(self, filename="expression_history.csv"):
        df = pd.DataFrame(self.history, columns=["Expression", "Result"])
        df.to_csv(filename, index=False)
        return f"Exported to {filename}"

    def export_history_to_excel(self, filename="expression_history.xlsx"):
        df = pd.DataFrame(self.history, columns=["Expression", "Result"])
        df.to_excel(filename, index=False)
        return f"Exported to {filename}"

    def show_vars(self):
        if not self.variables:
            return "No variables defined."

        print("Current Variables:")
        for var, val in self.variables.items():
            print(f"  {var} = {val}")


# --- Usage Example ---

if __name__ == "__main__":
    calc = ExpressionCalculator()
    calc.set_vars({'a': 5, 'b': 3, 'c': 2, 'd': 4, 'x': 10, 'y': 0})
    # Show variables
    calc.show_vars()

    test_expressions = [
        "a + b",                      # 5 + 3 = 8
        "a - c",                      # 5 - 2 = 3
        "a * b",                      # 5 * 3 = 15
        "a / d",                      # 5 / 4 = 1.25
        "(((a + b) - c) * 10) / d",   # (((5+3)-2)*10)/4 = 15
        "a + x - d",                  # 5 + 10 - 4 = 11
        "x / y",                      # 10 / 0 = Error
        "m + n",                      # Error: undefined vars
        "42 + 10",                    # Constant only = 52
        "(a + b) * (c + d)",          # (5+3)*(2+4) = 48
    ]

    for expr in test_expressions:
        result = calc.evaluate(expr)
        print(f"{expr} => {result}")

    # Export history
    print(calc.export_history_to_csv("expression_test_results.csv"))
#    print(calc.export_history_to_excel("expression_test_results.xlsx"))
    calc.set_vars({'base_price': 1000, 'discount': 0.1, 'tax': 50})
    result = calc.evaluate("base_price * (1 - discount) + tax")
    print(calc.show_history())
    print(f"{result}")
