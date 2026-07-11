def validate_medicine_data(medicine, times_per_day, days):
    
    if not medicine:
        return "Medicine name is required"
    
    if times_per_day <= 0:
        return "Times per day must be greater than 0"
    
    if days <= 0:
        return "Days must be greater than 0"

    return "Valid"
