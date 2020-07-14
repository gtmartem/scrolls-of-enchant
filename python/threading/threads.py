import logging
import queue
import random
import threading
import time
import concurrent.futures

# logging settings
fmt = "%(asctime)s: %(message)s"
logging.basicConfig(format=fmt, level=logging.INFO, datefmt="%H:%M:%S")


def thread_function(name, sleep):
    logging.info("Thread %s: starting", name)
    time.sleep(sleep)
    logging.info("Thread %s: finishing", name)


def thread_daemon_false():
    """
    Запуск треда с ожиданием завершения.
    :return: NoReturn
    """

    # start treading
    logging.info("\n-> thread_daemon_false")
    logging.info("Main: before creating thread")
    x = threading.Thread(target=thread_function, args=(1, 2))
    logging.info("Main: before running thread")
    x.start()
    logging.info("Main: wait for the thread to finish")
    # wait until thread finished
    x.join()
    logging.info("Main: all done")


def thread_daemon_true():
    """
    Запуск треда-демона без ожиданиея завершения.
    Тред-демон не завершится, так как при окончании работы скрипты (основного потока) -
    модуль treading шатдаунит все треды, созданные с daemon=True.
    :return: NoReturn
    """

    # start treading
    logging.info("\n-> thread_daemon_true")
    logging.info("Main: before creating daemon thread")
    x = threading.Thread(target=thread_function, args=(1, 2), daemon=True)
    logging.info("Main: before running daemon thread")
    x.start()
    logging.info("Main: wait 1 second for the daemon thread to finish")
    time.sleep(1)
    logging.info("Main: all done")


def complex_multithreading_example():
    """
    Запуск нескольких тредов наивным-сложным способом.
    :return: NoReturn
    """

    threads = []

    for index in range(3):
        logging.info("Main: create and start thread %d.", index)
        x = threading.Thread(target=thread_function, args=(index, 2))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        logging.info("Main: before joining thread %d.", index)
        thread.join()
        logging.info("Main: thread %d done", index)


def simple_multithreading_example():
    """
    Запуск нескольких тредов правильным способом.
    :return: NoReturn
    """

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(thread_function, range(3), [2] * 3)


def race_condition_example():
    """
    Пример иллюстрирует проблему "условия гонки".
    :return: NoReturn
    """

    class FakeDatabase:
        def __init__(self):
            self.value = 0

        def update(self, name):
            logging.info("Thread %s: starting update", name)
            local_copy = self.value
            local_copy += 1
            time.sleep(0.1)
            self.value = local_copy
            logging.info("Thread %s: finishing update", name)

    database = FakeDatabase()
    logging.info("Testing update. Starting value is %d.", database.value)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for index in range(2):
            executor.submit(database.update, index)
    logging.info("Testing update. Ending value is %d.", database.value)


def lock_example():
    """
    Пример иллюстрирует работу объекта Lock (MUTual EXclusion).
    :return: NoReturn
    """

    class FakeDatabase:
        def __init__(self):
            self.value = 0
            self._lock = threading.Lock()

        def locked_update(self, name):
            logging.info("Thread %s: starting update", name)
            logging.debug("Thread %s about to lock", name)
            with self._lock:
                logging.debug("Thread %s has lock", name)
                local_copy = self.value
                local_copy += 1
                time.sleep(0.1)
                self.value = local_copy
                logging.debug("Thread %s about to release lock", name)
            logging.debug("Thread %s after release", name)
            logging.info("Thread %s: finishing update", name)

    database = FakeDatabase()
    logging.info("Testing update. Starting value is %d.", database.value)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for index in range(2):
            executor.submit(database.locked_update, index)
    logging.info("Testing update. Ending value is %d.", database.value)


def pipeline_based_consumer_producer_example():
    """
    Пример иллюстрирует решение проблемы consumer-producer через pipeline, построенных на Lock.
    :return: NoReturn
    """

    class Pipeline:
        """Class to allow a single element pipeline between producer and consumer."""

        def __init__(self):
            self.message = 0
            self.producer_lock = threading.Lock()
            self.consumer_lock = threading.Lock()
            self.consumer_lock.acquire()

        def get_message(self, name):
            logging.debug("%s:about to acquire getlock", name)
            self.consumer_lock.acquire()
            logging.debug("%s:have getlock", name)
            message = self.message
            logging.debug("%s:about to release setlock", name)
            self.producer_lock.release()
            logging.debug("%s:setlock released", name)
            return message

        def set_message(self, message, name):
            logging.debug("%s:about to acquire setlock", name)
            self.producer_lock.acquire()
            logging.debug("%s:have setlock", name)
            self.message = message
            logging.debug("%s:about to release getlock", name)
            self.consumer_lock.release()
            logging.debug("%s:getlock released", name)

    SENTINEL = object()

    def producer(pipe):
        """Pretend we're getting a message from the network."""
        for index in range(10):
            message = random.randint(1, 101)
            logging.info("Producer got message: %s", message)
            pipe.set_message(message, "Producer")
        # Send a sentinel message to tell consumer we're done
        pipe.set_message(SENTINEL, "Producer")

    def consumer(pipe):
        """ Pretend we're saving a number in the database. """
        message = 0
        while message is not SENTINEL:
            message = pipe.get_message("Consumer")
            if message is not SENTINEL:
                logging.info("Consumer storing message: %s", message)

    pipeline = Pipeline()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(producer, pipeline)
        executor.submit(consumer, pipeline)


def queue_based_consumer_producer_example():
    """
    Пример иллюстрирует решение проблемы consumer-producer через queue.Queue.
    :return: NoReturn
    """

    def producer(pipe, ev):
        """Pretend we're getting a number from the network."""

        while not ev.is_set():
            message = random.randint(1, 101)
            logging.info("Producer got message: %s", message)
            pipe.put(message)
        logging.info("Producer received EXIT event. Exiting")

    def consumer(pipe, ev):
        """ Pretend we're saving a number in the database. """

        while not ev.is_set() or not pipe.empty():
            message = pipe.get()
            logging.info(
                "Consumer storing message: %s  (queue size=%s)",
                message,
                pipe.qsize(),
            )
        logging.info("Consumer received EXIT event. Exiting")

    pipeline = queue.Queue(maxsize=10)
    event = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(producer, pipeline, event)
        executor.submit(consumer, pipeline, event)
        time.sleep(0.1)
        logging.info("Main: about to set event")
        event.set()


if __name__ == "__main__":
    # thread_daemon_false()
    # thread_daemon_true()
    # complex_multithreading_example()
    # simple_multithreading_example()
    # race_condition_example()
    # lock_example()
    # pipeline_based_consumer_producer_example()
    queue_based_consumer_producer_example()
