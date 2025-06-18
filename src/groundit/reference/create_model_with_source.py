from typing import Type, get_origin, get_args
from pydantic import BaseModel, create_model, Field
from groundit.reference.models import FieldWithSource

# TODO: top level description is not preserved
def create_model_with_source(model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Dynamically creates a new Pydantic model for source tracking.

    This function transforms a given Pydantic model into a new one where each
    leaf field is replaced by a `FieldWithSource` generic model. This allows for
    tracking the original text (`source_quote`) for each extracted value while
    preserving the original field's type and description.

    - Leaf fields are converted to `FieldWithSource[OriginalType]`.
    - Nested Pydantic models are recursively transformed.
    - Lists and Unions are traversed to transform their inner types.
    - Field descriptions from the original model are preserved.

    Args:
        model: The original Pydantic model class to transform.

    Returns:
        A new Pydantic model class with source tracking capabilities.
    """
    
    def _transform_type(original_type: Type) -> Type:
        """Recursively transforms a type annotation."""
        origin = get_origin(original_type)

        if origin:  # Handles generic types like list, union, etc.
            args = get_args(original_type)
            transformed_args = tuple(_transform_type(arg) for arg in args)
            
            return origin[transformed_args]

        # Handle nested Pydantic models
        if isinstance(original_type, type) and issubclass(original_type, BaseModel):
            return create_model_with_source(original_type)

        # Handle NoneType for optional fields
        if original_type is type(None):
            return type(None)

        # Base case: for leaf fields, wrap in FieldWithSource
        return FieldWithSource[original_type]

    transformed_fields = {}
    for field_name, field_info in model.model_fields.items():
        new_type = _transform_type(field_info.annotation)
        
        # Create a new Field, preserving the original description and default value
        new_field = Field(
            description=field_info.description,
            default=field_info.default if not field_info.is_required() else ...
        )
        transformed_fields[field_name] = (new_type, new_field)

    source_model_name = f"{model.__name__}WithSource"
    return create_model(source_model_name, **transformed_fields, __base__=BaseModel)


def create_json_schema_with_source(json_schema: dict) -> dict:
    """
    Convert a JSON-Schema *produced from a Pydantic model* into a new schema
    that mirrors the behaviour of :pyfunc:`create_model_with_source` at the
    *schema level*.

    Each *leaf* value (i.e. a primitive type) is replaced by a reference to a
    ``FieldWithSource`` definition while preserving the original description
    and the overall structure of the document.  In addition, nested models
    declared in the ``$defs`` section are transformed recursively and the
    resulting definitions are stored under a new name with the suffix
    ``WithSource`` (e.g. ``Patient -> PatientWithSource``).

    Parameters
    ----------
    json_schema:
        A mapping that follows the JSON-Schema spec (as returned by
        ``BaseModel.model_json_schema()``).

    Returns
    -------
    dict
        The transformed schema.
    """
    # NOTE: The implementation purposefully mirrors the runtime transformation
    # executed by ``create_model_with_source`` but works directly on the JSON
    # representation to avoid the overhead of reconstructing Pydantic models.

    from copy import deepcopy
    from typing import Any, Mapping
    
    # We start with a deep-copy so that the input is never mutated in place.
    original_schema: dict[str, Any] = deepcopy(json_schema)

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    PRIMITIVE_JSON_TYPES = {"string", "integer", "number", "boolean"}

    # Mapping JSON type -> python type (needed to materialise the correct
    # FieldWithSource[T] schema via Pydantic).
    _json_to_py_type = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    # We lazily create FieldWithSource definitions as we encounter new leaf
    # types to avoid producing unused definitions.
    field_with_source_defs: dict[str, dict[str, Any]] = {}

    def _ensure_fws_definition(json_type: str) -> str:
        """Return the *definition key* for the given JSON primitive type.

        If the definition doesn't exist yet it will be generated on-the-fly
        using :pyclass:`FieldWithSource` so that the resulting schema matches
        exactly what Pydantic would have produced.
        """
        key = f"FieldWithSource_{_json_to_py_type[json_type].__name__}_"
        if key in field_with_source_defs:
            return key

        # Generate the schema for the specialised FieldWithSource model.
        fws_schema: dict[str, Any] = FieldWithSource[_json_to_py_type[json_type]].model_json_schema()

        # ``model_json_schema`` produces a *root* schema – we store it directly
        # under ``$defs`` with the key computed above.
        field_with_source_defs[key] = fws_schema
        return key

    # ------------------------------------------------------------------
    # First pass – transform *definitions* as they might be referenced from
    # multiple places in the main schema.
    # ------------------------------------------------------------------
    existing_defs: Mapping[str, Any] = original_schema.get("$defs", {})
    transformed_defs: dict[str, Any] = {}
    ref_remap: dict[str, str] = {}

    def _transform_definition(def_key: str, def_schema: dict[str, Any]) -> None:
        """Transform a definition in-place and register remapped ``$ref``s."""
        transformed_key = f"{def_key}WithSource"
        ref_remap[f"#/$defs/{def_key}"] = f"#/$defs/{transformed_key}"
        transformed_schema = _transform_schema(def_schema)
        # Update the title to reflect the WithSource suffix
        if isinstance(transformed_schema, dict) and "title" in transformed_schema:
            transformed_schema["title"] = transformed_key
        transformed_defs[transformed_key] = transformed_schema

    # ------------------------------------------------------------------
    # Core recursive transformation logic.
    # ------------------------------------------------------------------
    def _transform_schema(node: Any) -> Any:  # noqa: C901 – complex but clear
        """Recursively walk a JSON-schema fragment and apply the conversion."""
        if isinstance(node, dict):
            # Handle $ref early – replace if we have a remapping.
            if "$ref" in node:
                ref_value: str = node["$ref"]
                if ref_value in ref_remap:
                    # Copy to avoid mutating original reference node.
                    return {"$ref": ref_remap[ref_value]}
                return node  # Not a reference we generated – keep as-is.

            # Objects – dive into properties & defs.
            if node.get("type") == "object":
                # Recursively transform properties.
                props = node.get("properties", {})
                node["properties"] = {k: _transform_schema(v) for k, v in props.items()}

                # Also transform additional nested definitions if present.
                if "$defs" in node:
                    nested_defs = node["$defs"]
                    for k, v in list(nested_defs.items()):
                        _transform_definition(k, v)
                    # We *don't* keep nested defs inside the object – they'll be
                    # re-attached at the top level later on.
                    node.pop("$defs", None)

                return node

            # Arrays – walk the ``items`` schema.
            if node.get("type") == "array" and "items" in node:
                node["items"] = _transform_schema(node["items"])
                return node

            # Primitive leaf – replace with FieldWithSource ref.
            if "type" in node and node["type"] in PRIMITIVE_JSON_TYPES:
                json_type = node["type"]
                description = node.get("description")
                ref_key = _ensure_fws_definition(json_type)
                new_node: dict[str, Any] = {"$ref": f"#/$defs/{ref_key}"}
                if description is not None:
                    new_node["description"] = description
                return new_node

            # Composite constructs (anyOf/oneOf/allOf)
            for comb in ("anyOf", "oneOf", "allOf"):
                if comb in node:
                    node[comb] = [_transform_schema(sub) for sub in node[comb]]
            return node

        elif isinstance(node, list):
            return [_transform_schema(item) for item in node]
        else:
            return node

    # ------------------- Execute passes -------------------
    # Transform existing definitions first so that ``ref_remap`` is populated
    # for the main schema traversal.
    for key, schema_fragment in existing_defs.items():
        _transform_definition(key, schema_fragment)

    # Transform the *root* schema.
    transformed_root = _transform_schema(original_schema)

    # Adjust the root title to signal the new structure.
    if isinstance(transformed_root, dict) and "title" in transformed_root:
        transformed_root["title"] = f"{transformed_root['title']}WithSource"

    # ------------------------------------------------------------------
    # Assemble the final collection of definitions (original ones have been
    # transformed into ``transformed_defs``).
    # ------------------------------------------------------------------
    all_defs: dict[str, Any] = {}
    if transformed_defs:
        all_defs.update(transformed_defs)
    if field_with_source_defs:
        all_defs.update(field_with_source_defs)
    if all_defs:
        transformed_root["$defs"] = all_defs

    return transformed_root