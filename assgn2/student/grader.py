#!/usr/bin/env python

import random, util, collections
import graderUtil

grader = graderUtil.Grader()
submission = grader.load('submission')

try:
    import solution
    grader.addHiddenPart = grader.addBasicPart
    SEED = solution.SEED
    solution_exist = True
except ModuleNotFoundError:
    SEED = 42
    solution_exist = False

def test_correct(func_name, assertion=lambda pred: True, equal=lambda x, y: x == y):
    def test():
        pred = getattr(submission, func_name)()
        assert pred is None or assertion(pred)
        if solution_exist:
            answer = getattr(solution, func_name)()
            grader.requireIsTrue(equal(pred, answer))
    return test

def test_wrong(func_name, assertion=lambda pred: True):
    def test():
        pred = getattr(submission, func_name)()
        assert pred is None or assertion(pred)
        if solution_exist:
            answer = getattr(solution, func_name)()
            grader.requireIsTrue(pred != answer and pred is not None)
    return test

############################################################
# Problem 1

def test_1a_1():
    mdp1 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=10, peekCost=1)
    startState = mdp1.startState()
    preBustState = (6, None, (1, 1))
    postBustState = (11, None, None)

    mdp2 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=15, peekCost=1)
    preEmptyState = (11, None, (1,0))

    # Make sure the succAndProbReward function is implemented correctly.
    tests = [
        ([((1, None, (1, 2)), 0.5, 0), ((5, None, (2, 1)), 0.5, 0)], mdp1, startState, 'Take'),
        ([((0, 0, (2, 2)), 0.5, -1), ((0, 1, (2, 2)), 0.5, -1)], mdp1, startState, 'Peek'),
        ([((0, None, None), 1, 0)], mdp1, startState, 'Quit'),
        ([((7, None, (0, 1)), 0.5, 0), ((11, None, None), 0.5, 0)], mdp1, preBustState, 'Take'),
        ([], mdp1, postBustState, 'Take'),
        ([], mdp1, postBustState, 'Peek'),
        ([], mdp1, postBustState, 'Quit'),
        ([((12, None, None), 1, 12)], mdp2, preEmptyState, 'Take')
    ]
    for gold, mdp, state, action in tests:
        if not grader.requireIsEqual(gold,
                                     mdp.succAndProbReward(state, action)):
            print('   state: {}, action: {}'.format(state, action))
grader.addBasicPart('1a-1-basic', test_1a_1, 3, description="Basic test for succAndProbReward() that covers several edge cases.")

def test_1a_2():
    def solve(BlackjackMDP):
        mdp = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3,
                           threshold=40, peekCost=1)
        startState = mdp.startState()
        alg = util.ValueIteration()
        alg.solve(mdp, .0001)
        return alg.V[startState]

    pred = solve(submission.BlackjackMDP)
    if solution_exist:
        answer = solve(solution.BlackjackMDP)
        grader.requireIsTrue((abs(pred - answer) / answer) < 0.1)
        
grader.addHiddenPart('1a-2-hidden', test_1a_2, 2, description="Hidden test for ValueIteration. Run ValueIteration on BlackjackMDP, then test if V[startState] is correct.")

def get_test_1b(multiplier):
    if solution_exist:
        BlackjackMDP = solution.BlackjackMDP
    else:
        BlackjackMDP = submission.BlackjackMDP

    def solve(ValueIteration, *args):
        mdp = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3,
                                    threshold=40, peekCost=1)
        startState = mdp.startState()
        alg = ValueIteration()
        alg.solve(mdp, *args)
        return alg.V[startState]

    def test_1b():
        tm = graderUtil.TimeMeasure()

        for _ in range(3):
            tm.check()
            util_vi = solve(util.ValueIteration, .00001)
            util_vi_time = tm.elapsed()

            tm.check()
            pred_dp = solve(submission.ValueIterationDP)
            pred_dp_time = tm.elapsed()

            print('VI time: {} / DP time: {}'.format(util_vi_time, pred_dp_time))
            grader.requireIsTrue((abs(pred_dp - util_vi) / util_vi) < 0.00001)
            grader.requireIsTrue(pred_dp_time * multiplier < util_vi_time)

    return test_1b

grader.addHiddenPart('1b-1-hidden', get_test_1b(2), 3, description="Hidden test for ValueIterationDP. Run ValueIterationDP on BlackjackMDP, then test if V[startState] is correct and ValueIterationDP is faster than ValueIteration.")
grader.addHiddenPart('1b-2-hidden', get_test_1b(3), 2, description="Hidden test for ValueIterationDP. Run ValueIterationDP on BlackjackMDP, then test if V[startState] is correct and ValueIterationDP is faster than ValueIteration.")

############################################################

grader.grade()
