package com.carrental.controller;

import com.carrental.dto.CarResponse;
import com.carrental.dto.ReservationResponse;
import com.carrental.service.CarService;
import com.carrental.service.ReservationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/admin")
@RequiredArgsConstructor
@PreAuthorize("hasRole('ADMIN')")
@Tag(name = "Admin", description = "Admin-only endpoints")
public class AdminController {

    private final CarService carService;
    private final ReservationService reservationService;

    @GetMapping("/cars")
    @Operation(summary = "Get all cars (admin)", description = "Returns all cars including unavailable ones")
    public ResponseEntity<List<CarResponse>> getAllCars() {
        return ResponseEntity.ok(carService.getAllCars());
    }

    @GetMapping("/reservations")
    @Operation(summary = "Get all reservations (admin)", description = "Returns all reservations from all users")
    public ResponseEntity<List<ReservationResponse>> getAllReservations() {
        return ResponseEntity.ok(reservationService.getAllReservations());
    }
}
