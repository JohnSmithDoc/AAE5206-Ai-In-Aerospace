import math
import operator


def data_process():
    return 0


def get_distance(datapoint1, datapoint2):
    distance = 0
    # this is because we already know there are 4 number data for each datapoint
    for x in range(4):
        # calculate sqr
        distance += (datapoint1[x] - datapoint2[x]) ** 2

    final_distance = math.sqrt(distance)
    return final_distance


def get_K_neighbors(train_dataset, test_datapoint, K):
    ## 1. get all distances
    all_distances = []
    for i in range(len(train_dataset)):
        distance = get_distance(train_dataset[i], test_datapoint)
        # we get the distance between every train_datapoint and test_datapoint
        all_distances.append((train_dataset[i], distance))

    ### 2. sort all distances, list operator sort
    all_distances.sort(key=operator.itemgetter(1))

    ### 3. pickup the K nearest neighbours
    K_neighbors = []
    for k in range(K):
        # 把前K个距离最近的数据提取出来，最后通过判断这前K个最近数据中哪个标签出现的频率高或者说占的百分比大，就选哪个作为分类的预测
        K_neighbors.append(all_distances[k][0])

    return K_neighbors


def get_prediction(K_neighbors):
    # 建立一个空字典
    category_votes = {}
    for i in range(len(K_neighbors)):
        # -1表示从后往前列表的第一个元素，也即最后一个元素，同理-2则是倒数第二个元素，在这里就是'cat' 'dog'这类标签
        category = K_neighbors[i][-1]
        # 统计对应的标签以及出现次数，通过字典（键-值对）来统计和投票
        if category in category_votes:
            category_votes[category] += 1
        else:
            category_votes[category] = 1
    # 字典按值（即投票数）进行排序，降序排列
    sort_category_votes = sorted(category_votes.items(), key=operator.itemgetter(1), reverse=True)

    print(sort_category_votes)

    # 因为是降序排序，所以只用取最前面的即可，这里取[0][0]即把标签取出来了
    # 因为sorted对字典进行排序会返回一个列表，category_votes.items()表明返回一个包含字典键值对的列表，sorted()函数按照键进行排序，但其实还是按键的值进行排序
    # 类似的输出 [('dog', 2), ('cat', 1)]
    most_vote = sort_category_votes[0][0]
    return most_vote


def simple_case_study():
    ##### stage 1 :   data preparation
    train_dataset = [[2, 2, 2, 2, 'cat'],
                     [4, 4, 4, 4, 'dog'],
                     [1, 2, 1, 1, 'cat'],
                     [0, 3, 3, 3, 'dog']]

    test_datapoint = [5, 5, 5, 5]  # ? cat or dog

    ######  stage 2:  alrogithm
    K = 3
    K_neighbors = get_K_neighbors(train_dataset, test_datapoint, K)
    print('K neighbors:', K_neighbors)

    prediction = get_prediction(K_neighbors)
    print('final prediction of the test datapoint is:', prediction)


################################
if __name__ == "__main__":
    simple_case_study()
