from collections import deque

graph = {}
visits = {}
start = 'Симферополь'
end = 'Мурманск'


def add_graph(town_a, town_b, dist):
    s = {}
    if town_a in graph:
        s = graph[town_a]
    s[town_b] = dist
    graph[town_a] = s

    s = {}
    if town_b in graph:
        s = graph[town_b]
    s[town_a] = dist
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


def dfs(queue: deque, graph, check, res=[end]):
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
        dfs(queue, graph, check)
        if (res[0] in towns) and (check[left] == check[res[0]] - 1):
            res.insert(0, left)
            return res
    else:
        return 'Error'


def bfs(st, graph, check, res=[end]):
    if check[st] == 1:
        return False
    if st == end:
        return True
    check[st]=1
    towns = graph[st]
    for x in towns:
        if bfs(x, graph, check, res):
            res.insert(0, st)
            return res


read_f('test.txt')
fifo = deque([start])
print(bfs(start, graph, visits))
