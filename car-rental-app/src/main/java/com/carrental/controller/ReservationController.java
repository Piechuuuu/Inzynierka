package com.carrental.controller;

import com.carrental.dto.ReservationRequest;
import com.carrental.dto.ReservationResponse;
import com.carrental.security.CustomUserDetails;
import com.carrental.service.ReservationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/reservations")
@RequiredArgsConstructor
@Tag(name = "Reservations", description = "Reservation management endpoints")
public class ReservationController {

    private final ReservationService reservationService;

    @GetMapping
    @Operation(summary = "Get reservations", description = "User sees own reservations, Admin sees all - resolved by role enum")
    public ResponseEntity<List<ReservationResponse>> getReservations(
            @AuthenticationPrincipal CustomUserDetails userDetails) {
        return ResponseEntity.ok(
                reservationService.resolveReservationsByRole(userDetails.getAppUser())
        );
    }

    @PostMapping
    @Operation(summary = "Create reservation", description = "Create a new car reservation for the authenticated user")
    public ResponseEntity<ReservationResponse> createReservation(
            @Valid @RequestBody ReservationRequest request,
            @AuthenticationPrincipal CustomUserDetails userDetails) {
        return ResponseEntity.ok(
                reservationService.createReservation(request, userDetails.getAppUser())
        );
    }
}
