import time


def transformer_time_calculation_decorator(transformer_name: str):
    """
    Decorator for calculation duration of running transformer.

    :param transformer_name: name of pipelines transformer.
    :return: decorator function.
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            start_time = time.time()

            result = function(*args, **kwargs)

            end_time = time.time()
            duration = round(end_time - start_time, 2)
            print(f'{transformer_name} transformation ended, '
                  f'took {duration} seconds.')

            return result
        return wrapper
    return decorator
