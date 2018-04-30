# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg

# Return optimal action given the mdp solution.
def standardmdpaction(_, mdp_solution, s):
    a = mdp_solution.p(s)
    return a

def stdvalueiteration(mdp_data, r, vinit=None):
    """
    Run value iteration to solve a standard MDP.
    :param mdp_data:
    :param r:
    :param vinit:
    :return:
    """
    # Allocate initial value function & variables.
    diff = 1.0
    if vinit:
        vn = vinit
    else:
        vn = np.zeros([mdp_data.states, 1])

    # Perform value iteration.
    while diff >= 0.0001:
        vp = vn
        vn = np.amax(r + sum(mdp_data.sa_p.dot(vp(mdp_data.sa_s), 3)) * mdp_data.discount, axis=1)
        diff = np.amax(np.abs(vn - vp))
    # Return value function.
    v = vn
    return v


def stdpolicy(mdp_data, r, v):
    """
    Given reward and value functions, solve for q function and policy.
    :param mdp_data:
    :param r:
    :param v:
    :return:
    """

    # Compute Q function.
    q = r + np.sum(mdp_data.sa_p.dot(v(mdp_data.sa_s), 3)) * mdp_data.discount

    # Compute policy.
    p = np.amax(q, axis=1)
    return [q, p]


def standardmdpstep(mdp_data, _, s, a):
    """
    Take a single step with the specified action.
    :param mdp_data:
    :param s:
    :param a:
    :return:
    """

    # Random sample for stochastic step.
    r = np.random.rand(1)
    sm = 0
    for k in range(np.shape(mdp_data.sa_p)[2]):
        sm = sm + mdp_data.sa_p[s, a, k]
        if sm >= r:
            s = mdp_data.sa_s[s, a, k]
            return s

    # Should never reach here.
    print('ERROR: MDP data specifies transition distribution for state #i action #i that does not sum to 1!', s, a)
    s = -1

    return s


def standardmdpsolve(mdp_data, r):
    """
    Solve a standard MDP and return value function, Q function, and policy.
    :param mdp_data:
    :param r:
    :return:
    """

    # Run value iteration to compute the value function.
    v = stdvalueiteration(mdp_data, r)

    # Compute Q function and policy.
    [q, p] = stdpolicy(mdp_data, r, v)

    # Return solution.
    mdp_solution = {'v': v, 'q': q, 'p': p}
    return mdp_solution


def standardmdpfrequency(mdp_data, mdp_solution):
    """
    Compute the occupancy measure of the MDP given a policy.
    :param mdp_data:
    :param mdp_solution:
    :return:
    """

    # Build flow constraint matrix.
    # TODO: Sparce matrix
    # A = sparse([],[],[],mdp_data.states,mdp_data.states, mdp_data.states*np.shape(mdp_data.sa_p)[2])
    A = np.zeros([mdp_data.states, mdp_data.states])
    for s in range(mdp_data.states):
        A[s, s] = A[s, s] + 1
        a = mdp_solution.p[s]
        for k in range(np.shape(mdp_data.sa_p)[2]):
            sp = mdp_data.sa_s(s, a, k)
            A[sp, s] = A[sp, s] - mdp_data.discount * mdp_data.sa_p[s, a, k]

    # Solve linear system to get occupancy measure.
    x = linalg.solve(A, (1 / mdp_data.states) * np.ones([mdp_data.states, 1]))
    return x


def standardmdpcompare(p1, p2):
    """
    Compare two policies, return number of discrepancies.
    :param p1:
    :param p2:
    :return:
    """

    diff = np.count_nonzero(p1 != p2 & p1 != 0 & p2 != 0)
    return diff
