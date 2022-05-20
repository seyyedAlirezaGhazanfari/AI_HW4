class Elimination:
    @staticmethod
    def ordering(variables, bn):
        return variables

    def elimination_ask(self, x, e, bn):
        factors = []
        variables = Elimination.ordering(bn.keys(), bn)
        for var in variables:
            pass
