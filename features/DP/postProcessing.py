import numpy as np  
from sklearn.isotonic import IsotonicRegression

class PostProcessing:
    def __init__(self):
        pass

    @staticmethod
    def consistency(n, t):
        '''使用递增序列的consistency
            
            references
            ----------
            2009-ICDM-
        '''
        s = [0 for i in range(n)] # s[i]是t的前i项之和
        for i in range(0, n):
            if i == 0:
                s[0] = t[0]
            else:
                s[i] = s[i-1] + t[i]

        def M(i, j):
            if (i >= n) or (i > j): return 0
            res = s[j]
            if i != 0: res -= s[i-1]
            return 1.0 * res / (j - i + 1)

        # start
        J = [n-1]
        for k in range(n-1, -1, -1):
            jj, j = k, J[-1]
            while (len(J) > 0) and (jj+1 < n) and (M(jj+1, j) <= M(k, jj)):
                jj = j
                J.pop()
                if len(J) > 0: j = J[-1]
            J.append(jj)
        b = 0
        ret = [0 for i in range(n)] # 最终结果
        while len(J)>0:
            jj = J[-1]
            J.pop()
            for k in range(b, jj+1):
                ret[k] = M(b, jj)
            b = jj + 1
        return ret

    # reference: https://github.com/DPGraph/DPGraph/blob/91a87c95b39e46f4c2e78ba0edff9d8d9bf732d5/util.py#L91
    @staticmethod
    def pdf2Cdf(pdf):
        cdf = [0 for i in range(len(pdf))]
        if len(pdf) == 0:
            return cdf
        cdf[0] = pdf[0]
        for i in range(1, len(pdf)):
            cdf[i] = cdf[i-1] + pdf[i]
        return cdf

    @staticmethod
    def cdf2Pdf(cdf):
        pdf = [0 for i in range(len(cdf))]
        pdf[0] = cdf[0]
        for i in range(1, len(cdf)):
            pdf[i] = cdf[i] - cdf[i-1]
        return pdf


    @staticmethod
    def pdf2Rcdf(pdf):
        '''pdf 2 reversed cdf

            test ok:
                a = [1, 9, 4, 3, 4]
                b = PostProcessing.pdf2Rcdf(a)
                print('b = {}'.format(b)) b = [21, 20, 11, 7, 4]
        '''
        if len(pdf) == 0:
            return []
        rcdf = list(reversed(np.cumsum(list(reversed(pdf)))))
        return rcdf

    @staticmethod
    def rcdf2Pdf(rcdf):
        '''reversed cdf 2 pdf

            test ok:
                a = [1, 9, 4, 3, 4]
                b = PostProcessing.pdf2Rcdf(a)
                print('b = {}'.format(b))
                print('a = {}'.format(PostProcessing.rcdf2Pdf(b)))
                    b = [21, 20, 11, 7, 4]
                    a = [1, 9, 4, 3, 4]
        '''
        if len(rcdf) == 0:
            return []
        pdf = [0 for i in range(len(rcdf))]
        pdf[-1] = rcdf[-1]
        for i in range(len(rcdf)-2, -1, -1):
            pdf[i] = rcdf[i] - rcdf[i+1]
        return pdf

    @staticmethod
    def postProcessCdf(noisyCdf, totalCount):
        ir = IsotonicRegression(y_min=0, y_max=totalCount, increasing=True)
        cdf = ir.fit_transform(X=range(len(noisyCdf)), y=noisyCdf)
        return cdf

    @staticmethod
    def postProcessPdf(noisyPdf, totalCount):
        '''
            references
            ----------
            .. https://github.com/DPGraph/DPGraph/blob/91a87c95b39e46f4c2e78ba0edff9d8d9bf732d5/util.py#L91
        '''
        cdf = PostProcessing.pdf2Cdf(noisyPdf)
        cdf = PostProcessing.postProcessCdf(cdf, totalCount)
        pdf = PostProcessing.cdf2Pdf(cdf)
        return pdf


    '''Tianhao Wang
    '''
    @staticmethod
    def normSub(est_dist, n):
        '''
            est_dist: histogram
            n: 期望个数之和
            - https://github.com/vvv214/LDP_Protocols/blob/master/post-process/norm_sub.py
        '''
        estimates = np.copy(est_dist)
        while (np.fabs(sum(estimates) - n) > 1) or (estimates < 0).any():
            estimates[estimates < 0] = 0
            total = sum(estimates)
            mask = estimates > 0
            diff = (n - total) / sum(mask)
            estimates[mask] += diff
        return estimates

    @staticmethod
    def normSub_int(est_dist, n):
        '''整数的normSub
        '''
        estimates = np.copy(est_dist)
        addPos, subPos = 0, sum(est_dist)
        while (np.fabs(sum(estimates) - n) > 1) or (estimates < 0).any():
            estimates[estimates < 0] = 0
            total = sum(estimates)
            mask = estimates > 0

            nonZeroNum = sum(mask) # 非0的个数
            diff = (n - total) // nonZeroNum
            estimates[mask] += diff

            # 剩余的diff
            left = int(np.round((n - total) - diff * nonZeroNum))
            if left > 0:
                while left > 0:
                    if addPos >= nonZeroNum:
                        addPos = 0
                    estimates[addPos] += 1
                    addPos += 1
                    left -= 1
            elif left < 0:
                while left < 0:
                    if (subPos < 0) or (subPos >= nonZeroNum):
                        subPos = nonZeroNum - 1
                    estimates[subPos] -= 1
                    subPos -= 1
                    left += 1
            # print('tot, diff, left, n :', tot, diff, left, n)
        return estimates


    @staticmethod
    def CLS_total_count(hist, N):
        '''仿2020NDSS-CLS的思想, 将值降一个常数，保证\sum_i i*hist_i \geq N.
            缺点: 很容易聚集到0, 且减去的是小数
            hist <==> x
            min_{x'} \|x - x'\|_2
            s.t. sum_i x'_i \leq N
                x'_i \geq 0
        '''
        h = np.copy(hist)
    
        noisy_total_count = 0
        for i in range(len(h)):
            noisy_total_count += i * h[i]
        
        # left_total_count是D1中的值之和
        L, left_total_count = sum(hist) - hist[0], noisy_total_count
        if L == 0:
            return h
        # print('noisy_total_count = {}, L = {}, left_total_count = {}'.format(noisy_total_count, L, left_total_count))
        j = 0   
        for i in range(1, len(h)):
            delta = i - (left_total_count - N) / L # assume that (left_total_count >= N)
            # print('i = {}, delta = {}, left_total_count - N = {}, L = {}, h[i] = {}'.format(i, delta, left_total_count - N, L, h[i]))
            if (delta >= 0): # 表示已经可以得到非零值了；x取0还是取多少都不会影响；所以选择x=0
                j = i
                break
            
            h[0] += h[i] 
            L, left_total_count = L - h[i], left_total_count - h[i] * i
            h[i] = 0
        # 现在已经得到了哪些变成0，哪些是需要减去一个值的
        delta = int(np.ceil((left_total_count - N) / L))
        # print(3*3+4*4-(left_total_count - N) / L * 7)
        if delta > 0:
            for i in range(j, len(h)):
                h[i-delta] += h[i]
                h[i] = 0
        return h
    

    @staticmethod
    def scaling_total_count(hist, N):
        '''scaling, 输入需要保证\sum_i i*hist_i \geq N.
        '''
        h = np.copy(hist)
    
        noisy_total_count = 0
        for i in range(len(h)):
            noisy_total_count += i * h[i]
        
        if noisy_total_count <= N:
            return h

        for i in range(1, len(h)):
            # print('i = {}, i * N / noisy_total_count = {}'.format(i, i * N / noisy_total_count))
            v = h[i]
            h[int(np.floor(i * N / noisy_total_count))] += v
            h[i] -= v
        return h

if __name__ == '__main__':
    a = [1, 9, 4, 3, 4]
    b = PostProcessing.pdf2Rcdf(a)
    print('b = {}'.format(b))
    print('a = {}'.format(PostProcessing.rcdf2Pdf(b)))


    print(PostProcessing.consistency(5, [1, 9, 4, 3, 4])) # 1 5 5 5 5

    print(PostProcessing.postProcessCdf([1, 9, 4, 3, 4], 9))

    print(PostProcessing.postProcessPdf([1, 8, -5, -1, 1], 9))

    print(PostProcessing.CLS_total_count([1, 9, 4, 3, 4], 9))
    print(PostProcessing.scaling_total_count([1, 9, 4, 3, 4], 9))