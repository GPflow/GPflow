import inspect

ENABLE = True
SHAPE_ONLY = True


def print_locals(stack_index: int = 1) -> None:
    if not ENABLE:
        return

    stack = inspect.stack()
    frame_info = stack[stack_index]
    frame = frame_info.frame
    print("**************************************************")
    print("Locals of:", frame.f_code.co_name)
    print("**************************************************")

    sentinel = object()
    for name, value in frame.f_locals.items():
        if SHAPE_ONLY:
            shape = getattr(value, "shape", sentinel)
            if shape is not sentinel:
                value = shape
        print(name, value)
