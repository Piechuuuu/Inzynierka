package com.carrental.service;

import com.carrental.enums.DiscountType;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.Stream;

@Service
public class PricingService {

    private final BiFunction<BigDecimal, Long, BigDecimal> baseCostCalculator =
            (pricePerDay, days) -> pricePerDay.multiply(BigDecimal.valueOf(days));

    public BigDecimal calculateRentalCost(BigDecimal pricePerDay, LocalDate startDate, LocalDate endDate, boolean isYoungDriver) {
        long rentalDays = ChronoUnit.DAYS.between(startDate, endDate);

        BigDecimal baseCost = baseCostCalculator.apply(pricePerDay, rentalDays);

        List<DiscountType> applicableDiscounts = Stream.of(DiscountType.values())
                .filter(discount -> discount != DiscountType.YOUNG_DRIVER || isYoungDriver)
                .toList();

        BigDecimal finalMultiplier = applicableDiscounts.stream()
                .map(discount -> discount.getMultiplier(rentalDays))
                .reduce(BigDecimal.ONE, BigDecimal::multiply);

        return baseCost.multiply(finalMultiplier).setScale(2, RoundingMode.HALF_UP);
    }
}
