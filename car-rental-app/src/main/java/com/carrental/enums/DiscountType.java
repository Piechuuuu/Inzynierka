package com.carrental.enums;

import java.math.BigDecimal;
import java.util.function.Function;

public enum DiscountType {

    SHORT_RENTAL(days -> days >= 3 && days < 7
            ? BigDecimal.valueOf(0.80)
            : BigDecimal.ONE),

    LONG_RENTAL(days -> days >= 7
            ? BigDecimal.valueOf(0.60)
            : BigDecimal.ONE),

    YOUNG_DRIVER(days -> BigDecimal.valueOf(1.20));

    private final Function<Long, BigDecimal> multiplierFunction;

    DiscountType(Function<Long, BigDecimal> multiplierFunction) {
        this.multiplierFunction = multiplierFunction;
    }

    public BigDecimal getMultiplier(long rentalDays) {
        return multiplierFunction.apply(rentalDays);
    }
}
