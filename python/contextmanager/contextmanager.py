import asyncio
import logging
from logging import Logger
from pprint import pprint
from typing import Callable, Tuple

from async_generator import asynccontextmanager


@asynccontextmanager
async def retry_on_error(f: Callable, *args,
                         tries: int = 3, errors: Tuple = (),
                         logger: Logger = logging.getLogger(__name__),
                         **kwargs):
    error = None
    rv = None
    for _ in range(tries):
        try:
            # не используем yield тут, так как после отработки блока async with ошбика вернется сюда
            # и, если он совпадет со списком errors, то будет обработана, а не выброшена в стек.
            rv = await f(*args, **kwargs)
            error = None
            break
        except errors as err:
            error = err
            logger.warning(err)
    if error is not None:
        raise error
    yield rv


async def raiser(*args, **kwargs):
    pprint(args)
    pprint(kwargs)
    raise RuntimeError


async def raise_expected_error(*args, **kwargs):
    pprint(args)
    pprint(kwargs)
    raise ValueError


async def raise_on_first_run(*args, **kwargs):
    pprint(args)
    pprint(kwargs)
    if len(args[0]) == 0:
        args[0].append(1)
        raise ValueError
    return


async def main():
    try:
        pprint("raiser")
        async with retry_on_error(raiser, "args-a", kwarg_a="kwarg-a") as result:
            pprint("catch result")
    except RuntimeError:
        pprint("Handler RuntimeError")

    try:
        pprint("raise_expected_error")
        async with retry_on_error(raise_expected_error, "args-a", errors=(ValueError,), kwarg_a="kwarg-a") as result:
            pprint("catch result")
    except ValueError:
        pprint("Handler ValueError")

    try:
        pprint(raise_on_first_run)
        async with retry_on_error(raise_on_first_run, list(), errors=(ValueError,), kwarg_a="kwarg-a") as result:
            pprint("catch result")
    except ValueError:
        pprint("Handler ValueError")


    try:
        pprint("raise_on_first_run")
        async with retry_on_error(raise_on_first_run, list(), errors=(ValueError,), kwarg_a="kwarg-a") as result:
            pprint("catch result")
            raise ValueError
    except ValueError:
        pprint("Handler ValueError")


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())