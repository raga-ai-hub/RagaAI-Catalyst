export const isWithinDateRange = (date: string, startDate?: Date, endDate?: Date): boolean => {
    if (!startDate && !endDate) return true;
    
    const itemDate = new Date(date);
    const startOfDay = startDate ? new Date(startDate.setHours(0, 0, 0, 0)) : null;
    const endOfDay = endDate ? new Date(endDate.setHours(23, 59, 59, 999)) : null;
  
    const isAfterStart = startOfDay ? itemDate >= startOfDay : true;
    const isBeforeEnd = endOfDay ? itemDate <= endOfDay : true;
  
    return isAfterStart && isBeforeEnd;
  };