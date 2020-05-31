import math
class Measure(object):
    def __init__(self):
        pass


    @staticmethod
    def hits(origin, res):
        hitCount = {}
        for user in origin:
            items = origin[user]
            predicted = res[user]
            hitCount[user] = len(set(items).intersection(set(predicted)))
        return hitCount

    @staticmethod
    def rankingMeasure(origin, res, N):  # origin: true_item    res: 推荐item
        measure = []
        results = {}

        predicted = res
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set are not match!')
            exit(-1)
        hits = Measure.hits(origin, predicted)
        prec = Measure.precision(hits, N)
        indicators.append('Precision:' + str(prec))
        recall = Measure.recall(hits, origin)
        indicators.append('Recall:' + str(recall))
        F1 = Measure.F1(prec, recall)
        indicators.append('F1:' + str(F1))
        MAP = Measure.MAP(origin, predicted, N)
        indicators.append('MAP:' + str(MAP))
        NDCG = Measure.NDCG(origin, predicted, N)
        indicators.append('NDCG:' + str(NDCG))
        measure.append('Top ' + str(N))
        measure += indicators

        results['pre'] = prec
        results['rec'] = recall
        results['f1'] = F1
        results['MAP'] = MAP
        results['NDCG'] = NDCG

        return measure, results

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return round(float(prec) / (len(hits) * N),6)

    @staticmethod
    def MAP(origin, res, N):
        sum_prec = 0
        for user in res:
            hits = 0
            precision = 0
            for n, item in enumerate(res[user]):
                if item in origin[user]:
                    hits += 1
                    precision += hits / (n + 1.0)
            sum_prec += precision / (min(len(origin[user]), N) + 0.0)
        return sum_prec / (len(res))

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item in origin[user]:
                    DCG+= 1.0/math.log(n+2)
            for n, item in enumerate(origin[user]):
                IDCG+=1.0/math.log(n+2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / (len(res)),6)

    # @staticmethod
    # def AUC(origin, res, rawRes):
    #
    #     from random import choice
    #     sum_AUC = 0
    #     for user in origin:
    #         count = 0
    #         larger = 0
    #         itemList = rawRes[user].keys()
    #         for item in origin[user]:
    #             item2 = choice(itemList)
    #             count += 1
    #             try:
    #                 if rawRes[user][item] > rawRes[user][item2]:
    #                     larger += 1
    #             except KeyError:
    #                 count -= 1
    #         if count:
    #             sum_AUC += float(larger) / count
    #
    #     return float(sum_AUC) / len(origin)

    @staticmethod
    def recall(hits, origin):
        recallList = [float(hits[user]) / len(origin[user]) for user in hits]
        recall = sum(recallList) / float(len(recallList))
        return round(recall,6)

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall),6)
        else:
            return 0

    @staticmethod
    def ratingMeasure(origin, res):
        measure = []
        results = {}

        mae = Measure.MAE(origin, res)
        measure.append('MAE:' + str(mae))
        rmse = Measure.RMSE(origin, res)
        measure.append('RMSE:' + str(rmse))

        results['MAE'] = mae
        results['RMSE'] = rmse

        return measure, results

    @staticmethod
    def MAE(origin, res):
        error = 0
        count = 0
        for entry in list(zip(origin,res)):
            error+=abs(entry[0]-entry[1])
            count+=1
        if count==0:
            return error
        return float(error)/count

    @staticmethod
    def RMSE(origin, res):
        error = 0
        count = 0
        for entry in list(zip(origin, res)):
            error += (entry[0] - entry[1])**2
            count += 1
        if count==0:
            return error
        return math.sqrt(float(error)/count)




