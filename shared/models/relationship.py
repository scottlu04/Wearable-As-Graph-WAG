
"""A package containing the 'Relationship' model."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class Relationship:
    """A relationship between two entities. This is a generic relationship, and can be used to represent any type of relationship between any two entities."""
    id: str | None = None

    entity_1_name: str | None = None
    entity_1_description: str | None = None
    entity_1_id: str | None = None


    entity_2_name: str | None = None
    entity_2_description: str | None = None
    entity_2_id: str | None = None

    weight: float | None = 0
    """The edge weight. population model vs personalization, entropy ????"""

    description: str | None = None
    """A description of the relationship (optional)."""

    description_embedding: list[float] | None = None
    """The semantic embedding for the relationship description (optional)."""

    #reference to the web search results
    web_search_results: List[Any] | None = None
    """The web search results for the relationship (optional)."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entity instance to a dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        entity_1_name_key: str = 'entity_1_name',  # Fixed missing comma
        entity_1_id_key: str = 'entity_1_id',      # Added missing parameter
        entity_1_description_key: str = 'entity_1_description',
        entity_2_name_key: str = 'entity_2_name',  # Added missing parameter
        entity_2_id_key: str = 'entity_2_id',      # Added missing parameter
        entity_2_description_key: str = 'entity_2_description',
        description_key: str = "description",
        weight_key: str = "weight",
        web_search_results_key: str = "web_search_results",
    ) -> "Relationship":                            # Removed unused parameters
        """Create a new relationship from the dict data."""
        return Relationship(
            id=d[id_key],
            entity_1_name=d[entity_1_name_key],    # Updated to use correct fields
            entity_1_id=d[entity_1_id_key],
            entity_1_description=d.get(entity_1_description_key),
            entity_2_name=d[entity_2_name_key],
            entity_2_id=d[entity_2_id_key],
            entity_2_description=d.get(entity_2_description_key),
            description=d.get(description_key),
            weight=d.get(weight_key, 0),
            web_search_results=d.get(web_search_results_key),
        )
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entity instance to a dictionary."""
        return asdict(self)