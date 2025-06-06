
def add_source_spans(extraction_result: dict, source_text: str) -> dict:
    """
    Add source_span fields to an extraction result that contains source_quote fields.
    
    Args:
        extraction_result: JSON with source_quote fields
        source_text: The original source document text
        
    Returns:
        Enriched JSON with source_span fields added alongside source_quote fields
    """
    import copy
    
    def find_spans_recursive(data):
        """Recursively find spans for fields with source_quote"""
        
        if isinstance(data, dict):
            # Check if this is a FieldWithSource object
            if 'source_quote' in data:
                source_quote = data['source_quote']
                
                # Find the span of the source quote in the text
                start_index = source_text.find(source_quote)
                
                if start_index == -1:
                    print(f"⚠️  Warning: Could not find source quote in text: '{source_quote}'")
                    # Set a placeholder span
                    data['source_span'] = [-1, -1]
                else:
                    end_index = start_index + len(source_quote)
                    data['source_span'] = [start_index, end_index]
                    
            # Recursively process other fields in the dict
            for key, value in data.items():
                if key != 'source_span':  # Don't process the span we just added
                    find_spans_recursive(value)
                    
        elif isinstance(data, list):
            # Handle lists (like the 'given' field)
            for item in data:
                find_spans_recursive(item)
    
    # Create a deep copy to avoid modifying the original
    enriched_result = copy.deepcopy(extraction_result)
    
    # Add spans to the copy
    find_spans_recursive(enriched_result)
    
    return enriched_result