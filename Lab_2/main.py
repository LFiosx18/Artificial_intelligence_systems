import queue
from collections import deque

graph = {}
visits = {}
target_dist = {}
test = queue.PriorityQueue()
start = 'Симферополь'
end = 'Мурманск'


def add_graph(town_a, town_b, dist):
    s = {}
    if town_a in graph:
        s = graph[town_a]
    s[town_b] = int(dist)
    graph[town_a] = s

    s = {}
    if town_b in graph:
        s = graph[town_b]
    s[town_a] = int(dist)
    graph[town_b] = s

    visits[town_a] = -1
    visits[town_b] = -1


def read_f(file_name):
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            data = []
            for x in line.split(','):
                data.append(x.replace('\n', ''))
            add_graph(data[0], data[1], data[2])


def read_dist(file_name):
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            data = []
            for x in line.split(','):
                data.append(x.replace('\n', ''))
            target_dist[data[0]] = int(data[1])


def bfs(queue: deque, check, res=[end]):
    if len(queue) != 0:
        left = queue.popleft()
        if left == start:
            check[left] = 0
        towns = graph[left]
        for x in towns:
            if check[x] == -1:
                queue.append(x)
                check[x] = check[left] + 1
                if x == end:
                    res.insert(0, left)
                    return check
        bfs(queue, check)
        if (res[0] in towns) and (check[left] == check[res[0]] - 1):
            res.insert(0, left)
            return res
    else:
        return False


def dfs(st, check, res: list):
    if check[st] == 1:
        return False
    if st == end:
        return True
    check[st] = 1
    towns = graph[st]
    for x in towns:
        if dfs(x, check, res):
            res.insert(0, st)
            return res


def dls(st, check, res: list, lim=0):
    if check[st] == 1:
        return False
    if st == end:
        return True
    check[st] = 1
    if lim == 0:
        return False
    lim -= 1
    towns = graph[st]
    for x in towns:
        if dls(x, check, res, lim):
            res.insert(0, st)
            return res
        check[x] = -1


def iddfs(st, check, res: list):
    for i in range(len(visits)):
        ch = check.copy()
        result = dls(st, ch, res, i)
        if (result is not None) and (result is not False):
            print('addfl count: ' + str(i))
            return result


def bds(queueS: deque, queueE: deque, check, res=[]):
    left = queueS.popleft()
    right = queueE.popleft()
    if left == start:
        check[left] = 0
    if right == end:
        check[right] = 0
    if left in queueE:
        res.insert(0, left)
        return check
    if right in queueS:
        res.append(right)
        return check
    townsL = graph[left]
    for x in townsL:
        if x in queueE:
            res.insert(0, x)
            res.insert(0, left)
            return check
        if check[x] == -1:
            check[x] = check[left] + 1
            queueS.append(x)
    townsR = graph[right]
    for x in townsR:
        if x in queueS:
            res.append(right)
            res.append(x)
            return check
        if check[x] == -1:
            check[x] = check[right] + 1
            queueE.append(x)
    bds(queueS, queueE, check, res)
    if (res[0] in townsL) and (check[left] == check[res[0]] - 1):
        res.insert(0, left)
    if (res[len(res) - 1] in townsR) and (check[right] == check[res[len(res) - 1]] - 1):
        res.append(right)
    return res


def gs(queue: deque, check, res=[start]):
    point = queue.popleft()
    if point == end:
        return True
    if point == start:
        check[point] = 1
    towns = graph[point]
    min = 10000
    next = ''
    r = False
    for k, v in towns.items():
        if check[k] == -1:
            if v + target_dist[k] < min:
                min = v + target_dist[k]
                next = k
                r = True
            check[k] = 1
    if not r:
        res.pop(len(res) - 1)
        for x in towns:
            check[x] = -1
        gs(queue, check, res)
        return res

    queue.append(next)
    res.append(next)
    gs(queue, check, res)
    return res


def a_star():
    test.put((target_dist[start], start))
    from_town = {}
    while test.queue != 0:
        node = test.get()
        if node[1] == end:
            res = end
            result = [res]
            while True:
                min = 5000
                t = from_town[res]
                for k, v in t.items():
                    if v < min:
                        min = v
                        res = k
                result.insert(0, res)
                if res == start:
                    break
            print('The path has been found. The shortest distance is: ' + str(node[0]))
            return result
        towns = graph[node[1]]
        for k, v in towns.items():
            # if node[1] in from_town and k == from_town[node[1]]:
            #     continue
            test.put((node[0] - target_dist[node[1]] + v + target_dist[k], k))
            if k in from_town:
                t = from_town[k]
                if node[1] in t:
                    continue
                t[node[1]] = (node[0] - target_dist[node[1]] + v + target_dist[k])
                from_town[k] = t
            else:
                from_town[k] = {node[1]: node[0] - target_dist[node[1]] + v + target_dist[k]}
    return False


read_f('test.txt')
read_dist('target_dist.txt')

print('\nA*')
print(a_star())
