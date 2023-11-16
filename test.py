import torch

src = torch.load('src.pt')

def _crunch_src(src):
    
    num_years = 3 # Number of years with yearly average
    num_months = 8 # Number of months with monthly average
    num_weeks = 3*4 # Number of weeks with weakly average (#months * #weeks in a month)
    
    # Define start and end indixes
    end_idx_years = num_years * 365
    start_idx_months, end_idx_months = end_idx_years, num_years * 365 + num_months * 30
    start_idx_weeks, end_idx_weeks = end_idx_months, num_years * 365 + num_months * 30 + num_weeks * 7
    
    # Reshape the tensor to represent years, months, weeks, and days
    years_data = src[:end_idx_years].view(num_years, 365, -1).mean(dim=1)
    months_data = src[start_idx_months:end_idx_months].view(num_months, 30, -1).mean(dim=1)
    weeks_data = src[start_idx_weeks:end_idx_weeks].view(num_weeks, 7, -1).mean(dim=1)
    last_month_data = src[-30:].squeeze(0)  # Keep the last month as is

    print(years_data.shape, months_data.shape, weeks_data.shape, last_month_data.shape)
    # Concatenate the reduced data to create the final tensor
    src = torch.cat([years_data, months_data, weeks_data, last_month_data], dim=0)
    
    print(src)
    print(src.shape)
    
_crunch_src(src=src)