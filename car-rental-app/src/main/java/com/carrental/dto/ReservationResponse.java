package com.carrental.dto;

import lombok.*;

import java.math.BigDecimal;
import java.time.LocalDate;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ReservationResponse {

    private Long id;
    private String carBrand;
    private String carModel;
    private LocalDate startDate;
    private LocalDate endDate;
    private BigDecimal totalCost;
    private String username;
}
