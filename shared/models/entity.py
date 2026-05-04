

"""A package containing the 'Entity' model."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class Entity:
    """A protocol for an entity in the system."""
    id: str | None = None
    """ID of the entity."""
    name: str | None = None
    """Name of the entity."""
    type: str | None = None
    """Type of the entity (like sleep, activity, etc)."""

    description: str | None = None
    """Description of the entity."""

    
    range: str | None = None
    """Range of values, unit. (optional)"""

    recommendation: str | None = None
    """Recommendation for improvement. (optional)"""

    # data: Dict[str, List[Any]] = field(
    #     default_factory=lambda: {
    #         'timestamp': [],  # [id, data, path, device]
    #         'date': []        # [id, data, path, device]
    #     }
    # )
    dataSource: Dict[str, Any] = field(default_factory=lambda: {})
    """Data source of the entity.
    {
        "dataset_name": "string",
        "feature_name": "string",
        "description": "string",
        "range": "string",
        "unit": "string",
        "type": "timestamp or date"
    }
    """
    """Data associated with the modality (optional), either high level summary per day or timestamped."""

    weight: int | None = 1
    """Weight of the entity, used for sorting (optional). Higher weight indicates more important entity. This can be based on centrality or other metrics."""

    if_data_associated: bool | None = None
    """If data associated."""

    #embedding
    semantic_embedding: list[float] | None = None
    """The semantic (i.e. text) embedding of the entity (optional)."""

    name_embedding: list[float] | None = None
    """The name embedding of the entity (optional)."""

    graph_embedding: list[float] | None = None
    """The graph embedding of the entity, likely from node2vec (optional)."""

    #reference
    provided_name: str | None = None
    """Provided name of the entity."""

    provided_description: str | None = None
    """Provided description of the entity."""

    provided_range: str | None = None
    """Provided range of the entity."""

    umls_name: str | None = None
    """UMLS name of the entity."""

    cui: str | None = None
    """CUI of the entity if in UMLS."""

    umls_definition: str | None = None
    """UMLS definition of the entity."""

    raw_web_result: str | None = None
    """Raw web result."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entity instance to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        name_key: str = "name",
        type_key: str = "type",
        description_key: str = "description",
        range_key: str = "range",
        recommendation_key: str = "recommendation",
        dataSource_key: str = "dataSource",
        weight_key: str = "weight",
        if_data_associated_key: str = "if_data_associated",
        semantic_embedding_key: str = "semantic_embedding",
        name_embedding_key: str = "name_embedding",
        graph_embedding_key: str = "graph_embedding",
        provided_name_key: str = "provided_name",
        provided_description_key: str = "provided_description",
        provided_range_key: str = "provided_range",
        umls_name_key: str = "umls_name",
        cui_key: str = "cui",
        umls_definition_key: str = "umls_definition",
        raw_web_result_key: str = "raw_web_result",
    ) -> "Entity":
        """Create a new entity from the dict data."""
        # Handle the data field properly
        # data = {}  # Initialize data dictionary
        # data["timestamp"] = d["data"].get(timestamp_key,[])
        # data["date"] = d["data"].get(date_key,[])  

        return cls(
            id=d.get(id_key),
            name=d.get(name_key),
            type=d.get(type_key),
            description=d.get(description_key),
            range=d.get(range_key),
            recommendation=d.get(recommendation_key),
            dataSource=d.get(dataSource_key, {}),
            weight=d.get(weight_key),
            if_data_associated=d.get(if_data_associated_key),
            semantic_embedding=d.get(semantic_embedding_key),
            name_embedding=d.get(name_embedding_key),
            graph_embedding=d.get(graph_embedding_key),
            provided_name=d.get(provided_name_key),
            provided_description=d.get(provided_description_key),
            provided_range=d.get(provided_range_key),
            umls_name=d.get(umls_name_key),
            cui=d.get(cui_key),
            umls_definition=d.get(umls_definition_key),
            raw_web_result=d.get(raw_web_result_key),
        )

if __name__ == "__main__":
    # Create a sample dictionary with required and optional fields
    entity_data = {
    "id": "1",
    "title": "physical activity level",
    "short_id": "acitiviy_level",
    "type": "activity",
    "description": "Activity level can refer to a person's physical activity level, which is a way to quantify how much a person is active each day.\
        The numeric level of activity with: • 0: Non-wear • 1: Rest (MET level below 1.05) • \
        2: Inactive (MET level between 1.05 and 2) • 3: Low intensity activity (MET level between 2 and age/gender dependent limit) • \
        4: Medium intensity activity • 5: High intensity activity",
    # "name_embedding": [0.1, 0.2, 0.3],  # Example embedding vector
    # "description_embedding": [0.4, 0.5, 0.6],  # Example embedding vector
    # "graph_embedding": [0.7, 0.8, 0.9],  # Example embedding vector
    # "community": ["cardiology", "primary_care"],
    # "text_unit_ids": ["tu_001", "tu_002"],
    # "document_ids": ["doc_001", "doc_002"],
    # "degree": 5,  # This will be used as rank
    # "attributes": {
    #     "manufacturer": "HealthTech",
    #     "measurement_unit": "mmHg",
    #     "last_calibration": "2024-03-20"
    "data": {
            "oura": "data/oura/1234567890",
        }
    }
    entity = Entity.from_dict(entity_data)
    print(entity)
