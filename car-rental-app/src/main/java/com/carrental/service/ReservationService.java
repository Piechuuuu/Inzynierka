package com.carrental.service;

import com.carrental.dto.ReservationRequest;
import com.carrental.dto.ReservationResponse;
import com.carrental.entity.AppUser;
import com.carrental.entity.Car;
import com.carrental.entity.Reservation;
import com.carrental.enums.Role;
import com.carrental.repository.CarRepository;
import com.carrental.repository.ReservationRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ReservationService {

    private static final int YOUNG_DRIVER_AGE_THRESHOLD = 25;

    private final ReservationRepository reservationRepository;
    private final CarRepository carRepository;
    private final PricingService pricingService;

    public List<ReservationResponse> getReservationsForUser(AppUser user) {
        return reservationRepository.findByUserId(user.getId()).stream()
                .map(this::mapToResponse)
                .toList();
    }

    public List<ReservationResponse> getAllReservations() {
        return reservationRepository.findAll().stream()
                .map(this::mapToResponse)
                .toList();
    }

    public List<ReservationResponse> resolveReservationsByRole(AppUser user) {
        return Role.ROLE_ADMIN.equals(user.getRole())
                ? getAllReservations()
                : getReservationsForUser(user);
    }

    @Transactional
    public ReservationResponse createReservation(ReservationRequest request, AppUser user) {
        Car car = carRepository.findById(request.getCarId())
                .orElseThrow(() -> new IllegalArgumentException("Car not found with id: " + request.getCarId()));

        boolean isYoungDriver = user.getAge() < YOUNG_DRIVER_AGE_THRESHOLD;

        BigDecimal totalCost = pricingService.calculateRentalCost(
                car.getPricePerDay(),
                request.getStartDate(),
                request.getEndDate(),
                isYoungDriver
        );

        Reservation reservation = Reservation.builder()
                .car(car)
                .user(user)
                .startDate(request.getStartDate())
                .endDate(request.getEndDate())
                .totalCost(totalCost)
                .build();

        Reservation saved = reservationRepository.save(reservation);
        return mapToResponse(saved);
    }

    private ReservationResponse mapToResponse(Reservation reservation) {
        return ReservationResponse.builder()
                .id(reservation.getId())
                .carBrand(reservation.getCar().getBrand())
                .carModel(reservation.getCar().getModel())
                .startDate(reservation.getStartDate())
                .endDate(reservation.getEndDate())
                .totalCost(reservation.getTotalCost())
                .username(reservation.getUser().getUsername())
                .build();
    }
}
