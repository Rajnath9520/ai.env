import queue

from app.workers.broadcaster import _SENTINEL, MAX_QUEUE_SIZE, ExperimentBroadcaster


class TestExperimentBroadcaster:
    def test_publish_delivers_to_subscriber(self):
        broadcaster = ExperimentBroadcaster()
        q = broadcaster.subscribe(1)
        broadcaster.publish(1, {"episode": 1, "reward": 0.5})
        item = q.get_nowait()
        assert item == {"episode": 1, "reward": 0.5}

    def test_publish_to_unrelated_experiment_id_does_not_deliver(self):
        broadcaster = ExperimentBroadcaster()
        q = broadcaster.subscribe(1)
        broadcaster.publish(2, {"episode": 1})
        assert q.empty()

    def test_multiple_subscribers_all_receive(self):
        broadcaster = ExperimentBroadcaster()
        q1 = broadcaster.subscribe(1)
        q2 = broadcaster.subscribe(1)
        broadcaster.publish(1, {"episode": 5})
        assert q1.get_nowait() == {"episode": 5}
        assert q2.get_nowait() == {"episode": 5}

    def test_unsubscribe_stops_delivery(self):
        broadcaster = ExperimentBroadcaster()
        q = broadcaster.subscribe(1)
        broadcaster.unsubscribe(1, q)
        broadcaster.publish(1, {"episode": 1})
        assert q.empty()

    def test_close_sends_sentinel(self):
        broadcaster = ExperimentBroadcaster()
        q = broadcaster.subscribe(1)
        broadcaster.close(1)
        assert q.get_nowait() is _SENTINEL

    def test_queue_drops_oldest_when_full(self):
        broadcaster = ExperimentBroadcaster()
        q = broadcaster.subscribe(1)
        for i in range(MAX_QUEUE_SIZE + 10):
            broadcaster.publish(1, {"episode": i})

        assert q.qsize() <= MAX_QUEUE_SIZE
        items = []
        while True:
            try:
                items.append(q.get_nowait())
            except queue.Empty:
                break
        assert items[-1] == {"episode": MAX_QUEUE_SIZE + 9}

    def test_publish_with_no_subscribers_does_not_raise(self):
        broadcaster = ExperimentBroadcaster()
        broadcaster.publish(999, {"episode": 1})  # should not raise
