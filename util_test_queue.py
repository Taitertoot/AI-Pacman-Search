import util


def test_priority_queue():
    """
    'Test' util.PriorityQueue
    :return:
    """

    class TestObj(object):
        def __init__(self, v):
            self.state = v

    pq = util.PriorityQueue()

    # Create some TestObj's and push them onto the queue with varying priorities
    # The first tuple of letters are the 'state's for the items and the second
    # are the corresponding priorities (which are intentionally in non-ascending
    # order to show the PriorityQueue does its job sorting...)
    for test_obj_v, priority in zip(('e', 'f', 'g', 'h'), (1.3, 0.2, 0.5, 0.1)):
        pq.push(TestObj(test_obj_v), priority)

    print 'NOTE: when items are enqueued, the queue (the pq.heap list) contains a tuple:'
    print '    (<priority>, <heap_count>, <item>)'
    print 'where <priority> is the priority number'
    print '      <heap_count> is the heap\'s internal count of how many items have been added'
    print '      <item> is the item stored'

    print '\nCurrently in the queue (just iterating pq.heap list)'
    print '<index> <priority> <heap_count> <item> <item.state>'
    for i, (priority, count, item) in enumerate(pq.heap):
        print i, priority, count, item, item.state

    print '\nHere\'s the key test: call pq.find_and_extract to remove the item with state == \'g\' '
    popped_item = pq.find_and_extract(state='g')
    print 'found <item>, <item.state>:', popped_item, popped_item.state

    print '\nCurrently in the queue (just iterating pq.heap list)'
    print '<index> <priority> <heap_count> <item> <item.state>'
    for i, (priority, count, item) in enumerate(pq.heap):
        print i, priority, count, item, item.state

    print '\npush the item back on the queue with higher (i.e., smaller = 0.001) priority'
    pq.push(popped_item, 0.001)

    print '\nCurrently in the queue (just iterating pq.heap list)'
    print '<index> <priority> <heap_count> <item> <item.state>'
    for i, (priority, count, item) in enumerate(pq.heap):
        print i, priority, count, item, item.state

    print '\npop the queue, which should return the item with just pushed on, which should have highest priority'
    item = pq.pop()
    print 'popped:', item, item.state

    print '\nCurrently in the queue (just iterating pq.heap list)'
    print '<index> <priority> <heap_count> <item> <item.state>'
    for i, (priority, count, item) in enumerate(pq.heap):
        print i, priority, count, item, item.state

    print '\nDONE.'

test_priority_queue()
