package com.carrental;

import com.carrental.service.PricingService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.assertEquals;

class PricingServiceTest {

    private PricingService pricingService;

    @BeforeEach
    void setUp() {
        pricingService = new PricingService();
    }

    @Test
    @DisplayName("Should calculate base price for 2 days without discounts")
    void shouldCalculateBasePriceWithoutDiscount() {
        BigDecimal pricePerDay = BigDecimal.valueOf(100);
        LocalDate start = LocalDate.of(2025, 1, 1);
        LocalDate end = LocalDate.of(2025, 1, 3); // 2 days

        BigDecimal result = pricingService.calculateRentalCost(pricePerDay, start, end, false);

        // 2 days * 100 = 200, no discount
        assertEquals(new BigDecimal("200.00"), result);
    }

    @Test
    @DisplayName("Should apply 20% discount for 3+ days rental")
    void shouldApplyShortRentalDiscount() {
        BigDecimal pricePerDay = BigDecimal.valueOf(100);
        LocalDate start = LocalDate.of(2025, 1, 1);
        LocalDate end = LocalDate.of(2025, 1, 5); // 4 days

        BigDecimal result = pricingService.calculateRentalCost(pricePerDay, start, end, false);

        // 4 days * 100 = 400, * 0.80 = 320
        assertEquals(new BigDecimal("320.00"), result);
    }

    @Test
    @DisplayName("Should apply 40% discount for 7+ days rental")
    void shouldApplyLongRentalDiscount() {
        BigDecimal pricePerDay = BigDecimal.valueOf(100);
        LocalDate start = LocalDate.of(2025, 1, 1);
        LocalDate end = LocalDate.of(2025, 1, 8); // 7 days

        BigDecimal result = pricingService.calculateRentalCost(pricePerDay, start, end, false);

        // 7 days * 100 = 700, * 0.60 = 420
        assertEquals(new BigDecimal("420.00"), result);
    }

    @Test
    @DisplayName("Should apply young driver surcharge of 20%")
    void shouldApplyYoungDriverSurcharge() {
        BigDecimal pricePerDay = BigDecimal.valueOf(100);
        LocalDate start = LocalDate.of(2025, 1, 1);
        LocalDate end = LocalDate.of(2025, 1, 3); // 2 days

        BigDecimal result = pricingService.calculateRentalCost(pricePerDay, start, end, true);

        // 2 days * 100 = 200, * 1.20 (young driver) = 240
        assertEquals(new BigDecimal("240.00"), result);
    }

    @Test
    @DisplayName("Should combine long rental discount with young driver surcharge")
    void shouldCombineDiscountAndSurcharge() {
        BigDecimal pricePerDay = BigDecimal.valueOf(100);
        LocalDate start = LocalDate.of(2025, 1, 1);
        LocalDate end = LocalDate.of(2025, 1, 8); // 7 days

        BigDecimal result = pricingService.calculateRentalCost(pricePerDay, start, end, true);

        // 7 days * 100 = 700, * 0.60 (long rental) * 1.20 (young driver) = 504
        assertEquals(new BigDecimal("504.00"), result);
    }

    @Test
    @DisplayName("Should combine short rental discount with young driver surcharge")
    void shouldCombineShortDiscountAndSurcharge() {
        BigDecimal pricePerDay = BigDecimal.valueOf(200);
        LocalDate start = LocalDate.of(2025, 1, 1);
        LocalDate end = LocalDate.of(2025, 1, 4); // 3 days

        BigDecimal result = pricingService.calculateRentalCost(pricePerDay, start, end, true);

        // 3 days * 200 = 600, * 0.80 (short rental) * 1.20 (young driver) = 576
        assertEquals(new BigDecimal("576.00"), result);
    }
}
