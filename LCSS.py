""" LCSS Implementation """
# X, Y are the two trajectories
# C is the matrix storing all matching points


def LCSS(X, Y, matching_funct):
    m = len(X)
    n = len(Y)
    # (m+1) x (n+1) matrix
    C = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if matching_funct(X[i-1], Y[j-1]):
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    return C              # returns the whole matrix so that findSolution can be used on it


def findSolution(C, X, Y, i, j, matching_funct):     # first call should be with i == len(X) and j == len(Y)
    if i == 0 or j == 0:
        return []
    elif matching_funct(X[i-1], Y[j-1]):
    	subsolution = findSolution(C, X, Y, i-1, j-1, matching_funct)
    	subsolution.append(X[i-1])
        return subsolution
    else:
        if C[i][j-1] > C[i-1][j]:
            return findSolution(C, X, Y, i, j-1, matching_funct)
        else:
            return findSolution(C, X, Y, i-1, j, matching_funct)
