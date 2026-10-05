from typing import Type, TypeVar, Union

from pydantic import BaseModel
from src.state import CategoriesWithEvents

T = TypeVar("T", bound=BaseModel)


def ensure_pydantic_model(data: Union[dict, T], model_class: Type[T]) -> T:
    """Converts a dictionary to a Pydantic model instance if needed.
    If the data is already an instance of the model class, returns it as-is.

    Args:
        data: Either a dictionary or an instance of the Pydantic model
        model_class: The Pydantic model class to convert to

    Returns:
        An instance of the Pydantic model class

    Examples:
        # Convert dict to CategoriesWithEvents
        events = ensure_pydantic_model(some_dict, CategoriesWithEvents)

        # If already a model, returns as-is
        events = ensure_pydantic_model(existing_model, CategoriesWithEvents)
    """
    if isinstance(data, dict):
        return model_class(**data)
    elif isinstance(data, model_class):
        return data
    else:
        # Handle other cases - try to convert to dict first
        if hasattr(data, "__dict__"):
            return model_class(**data.__dict__)
        else:
            raise TypeError(f"Cannot convert {type(data)} to {model_class}")


# This function is needed because sometimes the object comes as a dict and then it's tricky to access the variables.
# There has to be a better way to do this in python, but this is the best I can come up with for now.
def ensure_categories_with_events(
    data: Union[dict, CategoriesWithEvents],
) -> "CategoriesWithEvents":
    """Specifically converts data to CategoriesWithEvents model."""
    return ensure_pydantic_model(data, CategoriesWithEvents)
