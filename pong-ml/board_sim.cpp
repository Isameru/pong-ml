
#include "pch.h"
#include "board_sim.h"

#include "config.h"

namespace pong
{
    namespace
    {
        vec2 to_vec2(cpVect v) { return {v.x, v.y}; }
    }

    BoardSim::BoardSim()
    {
        _space = cpSpaceNew();
        cpSpaceSetUserData(_space, this);

        cpBody* staticBody = cpSpaceGetStaticBody(_space);

        // Create the left and right board boundaries.
        //
        cpVect wallVertices[] = {
            { -0.5, -1.0 },
            { +0.5, -1.0 },
            { +0.5, +1.0 },
            { -0.5, +1.0 } };

        cpShape* wallShapes[] = {
            cpSpaceAddShape(_space, cpPolyShapeNew(staticBody, 4, wallVertices, cpTransform{1.0, 0.0, 0.0, 1.0, -1.0, 0.0}, 0.0)),
            cpSpaceAddShape(_space, cpPolyShapeNew(staticBody, 4, wallVertices, cpTransform{1.0, 0.0, 0.0, 1.0, +1.0, 0.0}, 0.0)) };
        for (auto* wallShape : wallShapes) {
            cpShapeSetElasticity(wallShape, cfg.wallsElasticity);
        }

        // Create the ball.
        //
        _ballBody = cpSpaceAddBody(_space, cpBodyNew(cfg.ballMass, cpMomentForCircle(cfg.ballMass, cfg.ballRadius, cfg.ballRadius, cpvzero)));
        cpShape* ballShape = cpSpaceAddShape(_space, cpCircleShapeNew(_ballBody, cfg.ballRadius, cpvzero));
        cpShapeSetCollisionType(ballShape, CollisionType::Ball);
        cpShapeSetElasticity(ballShape, cfg.ballElasticity);

        // Create the racquets.
        //
        for (int playerSlot = 0; playerSlot < 2; ++playerSlot)
        {
            auto& racquet = _racquets[playerSlot];
            const auto sideFactor = 2.0 * static_cast<double>(playerSlot) - 1.0;

            racquet.gripBody = cpSpaceAddBody(_space, cpBodyNew(cfg.racquetGripMass, cpMomentForCircle(cfg.racquetGripMass, cfg.racquetGripRadius, cfg.racquetGripRadius, cpvzero)));
            cpBodySetPosition(racquet.gripBody, cpVect{0.0, sideFactor * cfg.racquetToCenterDistance});
            racquet.gripShape = cpSpaceAddShape(_space, cpCircleShapeNew(racquet.gripBody, cfg.racquetGripRadius, cpvzero));
            cpShapeSetCollisionType(racquet.gripShape, CollisionType::Racquet);
            cpShapeSetElasticity(racquet.gripShape, cfg.racquetGripElasticity);

            cpVect racquetFaceVertices[] = {
                { -0.5 * sideFactor * cfg.racquetFaceDim.x, -0.5 * sideFactor * cfg.racquetFaceDim.y },
                { +0.5 * sideFactor * cfg.racquetFaceDim.x, -0.5 * sideFactor * cfg.racquetFaceDim.y },
                { +0.5 * sideFactor * cfg.racquetFaceDim.x, +0.5 * sideFactor * cfg.racquetFaceDim.y },
                { -0.5 * sideFactor * cfg.racquetFaceDim.x, +0.5 * sideFactor * cfg.racquetFaceDim.y } };

            racquet.faceBody = cpSpaceAddBody(_space, cpBodyNew(cfg.racquetFaceMass, cpMomentForPoly(cfg.racquetFaceMass, 4, racquetFaceVertices, cpvzero, 0.0)));
            cpBodySetPosition(racquet.faceBody, cpVect{0.0, sideFactor * (0.5 * cfg.racquetFaceDim.y + cfg.racquetToCenterDistance + cfg.racquetGripRadius)});
            racquet.faceShape = cpSpaceAddShape(_space, cpPolyShapeNew(racquet.faceBody, 4, racquetFaceVertices, cpTransformIdentity, 0.0));
            cpShapeSetCollisionType(racquet.faceShape, CollisionType::Racquet);
            cpShapeSetElasticity(racquet.faceShape, cfg.racquetFaceElasticity);

            cpSpaceAddConstraint(_space, cpPivotJointNew2(racquet.gripBody, racquet.faceBody, cpVect{0.0, 0.0}, cpVect{0.0, sideFactor * (-0.5 * cfg.racquetFaceDim.y - cfg.racquetGripRadius)}));
            cpSpaceAddConstraint(_space, cpGrooveJointNew(staticBody, racquet.gripBody, cpVect{-0.5f, sideFactor * cfg.racquetToCenterDistance}, cpVect{+0.5f, sideFactor * cfg.racquetToCenterDistance}, cpVect{0.0, 0.0}));
        }

        cpCollisionHandler* collisionHandler = cpSpaceAddCollisionHandler(_space, CollisionType::Ball, CollisionType::Racquet);
        collisionHandler->postSolveFunc = &BallRacquetCollisionPostSolveCallback;

        _state.match = -1;   // Shall be incremented to 0 with the first call to NewMatch().
    }

    BoardSim::~BoardSim()
    {
        if (_space != nullptr)
            cpSpaceDestroy(_space);
    }

    void BoardSim::NewMatch()
    {
        ++_state.match;
        _state.time = 0;
        _state.terminal = false;
        _state.winner = -1;

        cpSpaceSetGravity(_space, cpVect{0.0f, (_state.match % 2 == 0) ? -cfg.gravity : +cfg.gravity});

        ResetSpace();
        UpdateState();
    }

    void BoardSim::Step(std::array<uint8_t, 2> actions)
    {
        assert(_state.match >= 0 && "Call BoardSim::NewMatch() first");

        _state.scores = {0.0f, 0.0f};

        const double effectiveTimeStep = cfg.timeStep / static_cast<double>(cfg.subSteps);

        for (int subStep = 0; subStep < cfg.subSteps; ++subStep)
        {
            const auto ballPos = cpBodyGetPosition(_ballBody);
            const auto ballVel = cpBodyGetVelocity(_ballBody);
            _lastBallVelY = ballVel.y;
            const auto ballResultantForce = cpVect{-cfg.ballMotionResistance * ballVel.x, -cfg.ballMotionResistance * ballVel.y};
            cpBodyApplyForceAtWorldPoint(_ballBody, ballResultantForce, ballPos);

            for (int playerSlot = 0; playerSlot < 2; ++playerSlot)
            {
                auto& racquet = _racquets[playerSlot];
                const auto sideFactor = 2.0 * static_cast<double>(playerSlot) - 1.0;

                const auto action = actions[playerSlot];
                assert(action == 0 || action == 1);
                float move_x = static_cast<float>((2 * action) - 1);

                const auto racquetGripPos = cpBodyGetPosition(racquet.gripBody);
                const auto racquetGripResultantForce = cpVect{-sideFactor * cfg.racquetGripSteeringForce * move_x, 0.0f};
                cpBodyApplyForceAtWorldPoint(racquet.gripBody, racquetGripResultantForce, racquetGripPos);
                cpBodySetAngularVelocity(racquet.gripBody, 0.0);

                const auto racquetFacePos = cpBodyGetPosition(racquet.faceBody);
                const cpVect racquetFaceVel = cpBodyGetVelocity(racquet.faceBody);
                const auto racquetFaceResultantForce = cpVect{-cfg.racquetFaceMotionResistance * racquetFaceVel.x, -cfg.racquetFaceMotionResistance * racquetFaceVel.y};
                cpBodyApplyForceAtWorldPoint(racquet.faceBody, racquetFaceResultantForce, racquetFacePos);
            }

            cpSpaceStep(_space, effectiveTimeStep);
        }

        ++_state.time;

        UpdateState();
    }

    void BoardSim::UpdateState()
    {
        _state.gravity = cpSpaceGetGravity(_space).y;

        {
            cpVect pos = cpBodyGetPosition(_ballBody);
            _state.ball.pos = vec2{pos.x, pos.y};

            cpVect vel = cpBodyGetVelocity(_ballBody);
            _state.ball.vel = vec2{vel.x, vel.y};

            _state.ball.angularVel = cpBodyGetAngularVelocity(_ballBody);
        }

        for (int playerSlot = 0; playerSlot < 2; ++playerSlot)
        {
            _state.racquets[playerSlot].grip.pos = to_vec2(cpBodyGetPosition(_racquets[playerSlot].gripBody));
            _state.racquets[playerSlot].grip.vel = to_vec2(cpBodyGetVelocity(_racquets[playerSlot].gripBody));
            _state.racquets[playerSlot].face.pos = to_vec2(cpBodyGetPosition(_racquets[playerSlot].faceBody));
            _state.racquets[playerSlot].face.vel = to_vec2(cpBodyGetVelocity(_racquets[playerSlot].faceBody));
            _state.racquets[playerSlot].face.angle = static_cast<float>(cpBodyGetAngle(_racquets[playerSlot].faceBody));
            _state.racquets[playerSlot].face.angularVel = static_cast<float>(cpBodyGetAngularVelocity(_racquets[playerSlot].faceBody));
        }

        CheckTerminalCondition();
    }

    void BoardSim::CheckTerminalCondition()
    {
        if (_state.ball.pos.y < -1.0f) {
            _state.terminal = true;
            _state.winner = 1;
            _state.scores[1] = cfg.winScore;
        }
        else if (_state.ball.pos.y > 1.0f) {
            _state.terminal = true;
            _state.winner = 0;
            _state.scores[0] = cfg.winScore;
        }
        else if (_state.time > cfg.matchTickLimit) {
            // The match ends with a draw.
            _state.terminal = true;
        }
    }

    void BoardSim::ResetSpace()
    {
        cpBodySetPosition(_ballBody, cpvzero);
        cpBodySetVelocity(_ballBody, cpvzero);
        cpBodySetAngle(_ballBody, 0.0);
        cpBodySetAngularVelocity(_ballBody, 0.0);

        for (int playerSlot = 0; playerSlot < 2; ++playerSlot)
        {
            auto& racquet = _racquets[playerSlot];
            const auto sideFactor = 2.0 * static_cast<double>(playerSlot) - 1.0;

            cpBodySetPosition(racquet.gripBody, cpVect{0.0, sideFactor * cfg.racquetToCenterDistance});
            cpBodySetVelocity(racquet.gripBody, cpvzero);
            cpBodySetAngle(racquet.gripBody, 0.0);
            cpBodySetAngularVelocity(racquet.gripBody, 0.0);
            cpBodySetPosition(racquet.faceBody, cpVect{0.0, sideFactor * (0.5 * cfg.racquetFaceDim.y + cfg.racquetToCenterDistance + cfg.racquetGripRadius)});
            cpBodySetVelocity(racquet.faceBody, cpvzero);
            cpBodySetAngle(racquet.faceBody, 0.0);
            cpBodySetAngularVelocity(racquet.faceBody, 0.0);
        }
    }

    void BoardSim::BallRacquetCollisionPostSolveCallback(cpArbiter* arbiter, cpSpace* space, void* data)
    {
        reinterpret_cast<BoardSim*>(cpSpaceGetUserData(space))->OnBallRacquetCollisionPostSolve(arbiter);
    }

    void BoardSim::OnBallRacquetCollisionPostSolve(cpArbiter* arbiter)
    {
        cpShape* ballShape {};
        cpShape* racquetShape {};
        cpArbiterGetShapes(arbiter, &ballShape, &racquetShape);

        if (racquetShape == _racquets[0].faceShape || racquetShape == _racquets[0].gripShape)
        {
            if (cpSpaceGetGravity(_space).y < 0.0f) {
                cpSpaceSetGravity(_space, cpVect{0.0f, cfg.gravity});
                _state.scores[0] += cfg.ballInterceptionScore;
            }

            _state.scores[0] += +cfg.ballSendingScoreToSpeed * (cpBodyGetVelocity(_ballBody).y - _lastBallVelY);
        }
        else if (racquetShape == _racquets[1].faceShape || racquetShape == _racquets[1].gripShape)
        {
            if (cpSpaceGetGravity(_space).y > 0.0f) {
                cpSpaceSetGravity(_space, cpVect{0.0f, -cfg.gravity});
                _state.scores[1] += cfg.ballInterceptionScore;
            }

            _state.scores[1] += -cfg.ballSendingScoreToSpeed * (cpBodyGetVelocity(_ballBody).y - _lastBallVelY);
        }
        else assert(false);
    }

} // namespace pong
