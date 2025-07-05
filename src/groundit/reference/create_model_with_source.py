from typing import Type, get_origin, get_args, Literal, Any
from pydantic import BaseModel, create_model, Field
from groundit.reference.models import FieldWithSource


def create_model_with_source(
    model: Type[BaseModel], enrichment_class: Type[FieldWithSource] = FieldWithSource
) -> Type[BaseModel]:
    """
    Dynamically creates a new Pydantic model for source tracking.

    This function transforms a given Pydantic model into a new one where each
    leaf field is replaced by an enrichment class (e.g., `FieldWithSource` or
    `FieldWithSourceAndConfidence`). This allows for tracking the original text
    (`source_quote`) for each extracted value while preserving the original
    field's type and description.

    - Leaf fields are converted to `enrichment_class[OriginalType]`.
    - Nested Pydantic models are recursively transformed.
    - Lists and Unions are traversed to transform their inner types.
    - Field descriptions from the original model are preserved.

    Args:
        model: The original Pydantic model class to transform.
        enrichment_class: The enrichment class to use for leaf fields.
                         Defaults to FieldWithSource for backward compatibility.

    Returns:
        A new Pydantic model class with source tracking capabilities.
    """

    def _transform_type(original_type: Type) -> Any:
        """Recursively transforms a type annotation."""
        origin = get_origin(original_type)

        if origin:  # Handles generic types like list, union, etc.
            # Special case: Literal types should be wrapped as a whole
            if origin is Literal:
                return enrichment_class[original_type]  # type: ignore[misc]

            args = get_args(original_type)
            transformed_args = tuple(_transform_type(arg) for arg in args)

            return origin[transformed_args]

        # Handle nested Pydantic models
        if isinstance(original_type, type) and issubclass(original_type, BaseModel):
            return create_model_with_source(original_type, enrichment_class)

        # Handle NoneType for optional fields
        if original_type is type(None):
            return type(None)

        # Base case: for leaf fields, wrap in enrichment_class
        return enrichment_class[original_type]

    transformed_fields = {}
    for field_name, field_info in model.model_fields.items():
        new_type = _transform_type(field_info.annotation)

        # Create a new Field, preserving the original description and default value
        new_field = Field(
            description=field_info.description,
            default=field_info.default if not field_info.is_required() else ...,
        )
        transformed_fields[field_name] = (new_type, new_field)

    enrichment_suffix = enrichment_class.__name__.replace("FieldWith", "With")
    source_model_name = f"{model.__name__}{enrichment_suffix}"
    return create_model(source_model_name, **transformed_fields, __base__=BaseModel)


def create_json_schema_with_source(
    json_schema: dict, enrichment_class: Type[FieldWithSource] = FieldWithSource
) -> dict:
    """
    Convert a JSON-Schema *produced from a Pydantic model* into a new schema
    that mirrors the behaviour of :pyfunc:`create_model_with_source` at the
    *schema level*.

    Each *leaf* value (i.e. a primitive type) is replaced by a reference to an
    enrichment class definition (e.g., ``FieldWithSource`` or
    ``FieldWithSourceAndConfidence``) while preserving the original description
    and the overall structure of the document.  In addition, nested models
    declared in the ``$defs`` section are transformed recursively and the
    resulting definitions are stored under a new name with the appropriate suffix
    (e.g. ``Patient -> PatientWithSource`` or ``Patient -> PatientWithSourceAndConfidence``).

    Parameters
    ----------
    json_schema:
        A mapping that follows the JSON-Schema spec (as returned by
        ``BaseModel.model_json_schema()``).
    enrichment_class:
        The enrichment class to use for leaf fields. Defaults to FieldWithSource
        for backward compatibility.

    Returns
    -------
    dict
        The transformed schema.
    """
    # NOTE: The implementation purposefully mirrors the runtime transformation
    # executed by ``create_model_with_source`` but works directly on the JSON
    # representation to avoid the overhead of reconstructing Pydantic models.

    from copy import deepcopy
    from typing import Mapping

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

    # We lazily create enrichment class definitions as we encounter new leaf
    # types to avoid producing unused definitions.
    enrichment_defs: dict[str, dict[str, Any]] = {}

    def _ensure_enrichment_definition(json_type: str) -> str:
        """Return the *definition key* for the given JSON primitive type.

        If the definition doesn't exist yet it will be generated on-the-fly
        using the specified enrichment class so that the resulting schema matches
        exactly what Pydantic would have produced.
        """
        enrichment_name = enrichment_class.__name__
        key = f"{enrichment_name}_{_json_to_py_type[json_type].__name__}_"
        if key in enrichment_defs:
            return key

        # Generate the schema for the specialised enrichment class model.
        enrichment_schema: dict[str, Any] = enrichment_class[
            _json_to_py_type[json_type]  # type: ignore[misc]
        ].model_json_schema()

        # ``model_json_schema`` produces a *root* schema – we store it directly
        # under ``$defs`` with the key computed above.
        enrichment_defs[key] = enrichment_schema
        return key

    def _ensure_enrichment_literal_definition(enum_values: list, json_type: str) -> str:
        """Return the *definition key* for a Literal type with enum values.

        Creates an enrichment class definition that preserves the enum constraint
        in the value field, matching the behavior of create_model_with_source.
        """
        # Create a temporary Literal type to generate the correct schema
        from typing import Literal

        # Generate enrichment class schema for this specific Literal type
        literal_type = Literal[tuple(enum_values)]
        enrichment_schema: dict[str, Any] = enrichment_class[
            literal_type
        ].model_json_schema()

        # Create the key manually using the same pattern Pydantic uses
        # Pattern observation:
        # - Integers: "1__2__3__4__5__" (double underscores between)
        # - Strings: "__active____inactive____pending___" (underscores around each)
        key_parts = []
        for val in enum_values:
            if isinstance(val, str):
                key_parts.append(f"_{val}_")
            else:
                key_parts.append(str(val))

        if all(isinstance(v, str) for v in enum_values):
            # All strings: join parts (already wrapped with _) with double underscores, add trailing underscore
            key_suffix = "__".join(key_parts) + "__"
        else:
            # Non-strings (integers, etc.): join with double underscores, add trailing underscore
            key_suffix = "__".join(key_parts) + "__"

        enrichment_name = enrichment_class.__name__
        key = f"{enrichment_name}_Literal_{key_suffix}"

        if key in enrichment_defs:
            return key

        # Store the schema (should be root level, not nested)
        enrichment_defs[key] = enrichment_schema
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
        enrichment_suffix = enrichment_class.__name__.replace("FieldWith", "With")
        transformed_key = f"{def_key}{enrichment_suffix}"
        ref_remap[f"#/$defs/{def_key}"] = f"#/$defs/{transformed_key}"
        transformed_schema = _transform_schema(def_schema)
        # Update the title to reflect the enrichment suffix
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

            # Handle enum fields (Literal types in Pydantic) before primitive types
            if "enum" in node:
                enum_values = node["enum"]
                json_type = node.get("type")  # May be None for mixed type enums
                description = node.get("description")
                ref_key = _ensure_enrichment_literal_definition(enum_values, json_type)
                new_node: dict[str, Any] = {"$ref": f"#/$defs/{ref_key}"}
                if description is not None:
                    new_node["description"] = description
                return new_node

            # Primitive leaf – replace with enrichment class ref.
            if "type" in node and node["type"] in PRIMITIVE_JSON_TYPES:
                json_type = node["type"]
                description = node.get("description")
                ref_key = _ensure_enrichment_definition(json_type)
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
        enrichment_suffix = enrichment_class.__name__.replace("FieldWith", "With")
        transformed_root["title"] = f"{transformed_root['title']}{enrichment_suffix}"

    # ------------------------------------------------------------------
    # Assemble the final collection of definitions (original ones have been
    # transformed into ``transformed_defs``).
    # ------------------------------------------------------------------
    all_defs: dict[str, Any] = {}
    if transformed_defs:
        all_defs.update(transformed_defs)
    if enrichment_defs:
        all_defs.update(enrichment_defs)
    if all_defs:
        transformed_root["$defs"] = all_defs

    return transformed_root
