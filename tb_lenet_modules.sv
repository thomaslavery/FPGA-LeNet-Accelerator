// Testbench — exercises mac_unit, conv_hw, pool, relu, dense, and full LeNet
`timescale 1ns/1ps

module tb_lenet_modules;

    // -----------------------------------------------------------------------
    // Clock / Reset
    // -----------------------------------------------------------------------
    logic clk, rst_n;
    initial clk = 0;
    always #5 clk = ~clk;   // 100 MHz

    initial begin
        rst_n = 0;
        #20 rst_n = 1;
    end

    // =======================================================================
    // 1. MAC UNIT TEST
    // =======================================================================
    localparam K = 3;                          // 3x3 kernel for quick test
    localparam MAC_ACC = 8*2 + $clog2(K*K);

    logic signed [7:0]          mac_in  [0:K*K-1];
    logic signed [7:0]          mac_wt  [0:K*K-1];
    logic signed [MAC_ACC-1:0]  mac_out;
    logic                       mac_en, mac_valid;

    mac_unit #(.DATA_WIDTH(8), .KERNEL_SIZE(K)) u_mac (
        .clk     (clk),
        .rst_n   (rst_n),
        .en      (mac_en),
        .in_data (mac_in),
        .weight  (mac_wt),
        .result  (mac_out),
        .valid   (mac_valid)
    );

    // =======================================================================
    // 2. CONV_HW TEST  (4x4 input, 3x3 kernel -> 2x2 output)
    // =======================================================================
    localparam IMG_H = 4, IMG_W = 4, KS = 3;
    localparam OUT_H = IMG_H - KS + 1;   // 2
    localparam OUT_W = IMG_W - KS + 1;   // 2
    localparam CACC  = 8*2 + $clog2(KS*KS);

    logic signed [7:0]         fm   [0:0][0:IMG_H-1][0:IMG_W-1];
    logic signed [7:0]         wt   [0:0][0:KS-1][0:KS-1];
    logic signed [CACC-1:0]    conv_result [0:OUT_H-1][0:OUT_W-1];
    logic                      conv_start, conv_ready;

    conv_hw #(
        .DATA_WIDTH  (8),
        .KERNEL_SIZE (KS),
        .IMG_H       (IMG_H),
        .IMG_W       (IMG_W),
        .IN_CH       (1)
    ) u_conv (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (conv_start),
        .input_fm (fm),
        .weight   (wt),
        .result   (conv_result),
        .ready    (conv_ready)
    );

    // =======================================================================
    // 3. POOL TEST  (4x4 input -> 2x2 output, AVG mode)
    // =======================================================================
    logic signed [7:0] pool_in  [0:3][0:3];
    logic signed [7:0] pool_out [0:1][0:1];
    logic              pool_start, pool_valid;

    pool #(
        .DATA_WIDTH (8),
        .POOL_SIZE  (2),
        .IN_H       (4),
        .IN_W       (4),
        .MODE       (0)    // 0 = AVG
    ) u_pool (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (pool_start),
        .i      (pool_in),
        .result (pool_out),
        .valid  (pool_valid)
    );

    // =======================================================================
    // 4. RELU TEST  (5 elements)
    // =======================================================================
    localparam RELU_N = 5;

    logic signed [7:0] relu_in  [0:RELU_N-1];
    logic signed [7:0] relu_out [0:RELU_N-1];
    logic              relu_start, relu_valid;

    relu #(
        .DATA_WIDTH   (8),
        .NUM_ELEMENTS (RELU_N)
    ) u_relu (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (relu_start),
        .in_data  (relu_in),
        .out_data (relu_out),
        .valid    (relu_valid)
    );

    // =======================================================================
    // 5. DENSE TEST  (3 -> 2, small network)
    // =======================================================================
    localparam DENSE_IN  = 3;
    localparam DENSE_OUT = 2;
    localparam DENSE_ACC = 8*2 + $clog2(DENSE_IN);

    logic signed [7:0]            dense_in  [0:DENSE_IN-1];
    logic signed [7:0]            dense_w   [0:DENSE_OUT-1][0:DENSE_IN-1];
    logic signed [DENSE_ACC-1:0]  dense_b   [0:DENSE_OUT-1];
    logic signed [DENSE_ACC-1:0]  dense_out [0:DENSE_OUT-1];
    logic                         dense_start, dense_valid;

    dense #(
        .DATA_WIDTH  (8),
        .IN_FEATURES (DENSE_IN),
        .OUT_FEATURES(DENSE_OUT)
    ) u_dense (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (dense_start),
        .in_data (dense_in),
        .weight  (dense_w),
        .bias    (dense_b),
        .out_data(dense_out),
        .valid   (dense_valid)
    );

    // =======================================================================
    // 6. FULL LENET FORWARD PASS TEST
    // =======================================================================
    localparam LDW       = 8;  // DATA_WIDTH for LeNet
    localparam FC5_ACC_W = LDW * 2 + $clog2(400);
    localparam FC6_ACC_W = LDW * 2 + $clog2(120);
    localparam FC7_ACC_W = LDW * 2 + $clog2(84);

    logic signed [LDW-1:0]       lenet_image     [0:31][0:31];
    logic signed [LDW-1:0]       lenet_c1_w      [0:5][0:0][0:4][0:4];
    logic signed [LDW-1:0]       lenet_c3_w      [0:15][0:5][0:4][0:4];
    logic signed [LDW-1:0]       lenet_fc5_w     [0:119][0:399];
    logic signed [FC5_ACC_W-1:0] lenet_fc5_b     [0:119];
    logic signed [LDW-1:0]       lenet_fc6_w     [0:83][0:119];
    logic signed [FC6_ACC_W-1:0] lenet_fc6_b     [0:83];
    logic signed [LDW-1:0]       lenet_fc7_w     [0:9][0:83];
    logic signed [FC7_ACC_W-1:0] lenet_fc7_b     [0:9];
    logic signed [FC7_ACC_W-1:0] lenet_scores    [0:9];
    logic                        lenet_start, lenet_done;

    lenet #(
        .DATA_WIDTH(LDW)
    ) u_lenet (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (lenet_start),
        .image      (lenet_image),
        .c1_weight  (lenet_c1_w),
        .c3_weight  (lenet_c3_w),
        .fc5_weight (lenet_fc5_w),
        .fc5_bias   (lenet_fc5_b),
        .fc6_weight (lenet_fc6_w),
        .fc6_bias   (lenet_fc6_b),
        .fc7_weight (lenet_fc7_w),
        .fc7_bias   (lenet_fc7_b),
        .scores     (lenet_scores),
        .done       (lenet_done)
    );

    // =======================================================================
    // Stimulus
    // =======================================================================
    integer r, c, pass_count, fail_count;
    logic all_correct;

    initial begin
        mac_en      = 0;
        conv_start  = 0;
        pool_start  = 0;
        relu_start  = 0;
        dense_start = 0;
        lenet_start = 0;
        pass_count  = 0;
        fail_count  = 0;

        @(posedge rst_n); #10;

        // ------------------------------------------------------------------
        // TEST 1: MAC unit — 3x3 window, all inputs=2, all weights=3
        //         Expected: 2*3*9 = 54
        // ------------------------------------------------------------------
        $display("\n=== TEST 1: MAC Unit ===");
        foreach (mac_in[i]) mac_in[i] = 8'sd2;
        foreach (mac_wt[i]) mac_wt[i] = 8'sd3;

        @(posedge clk); #1;
        mac_en = 1;
        @(posedge clk); #1;
        // MAC computes in 1 cycle — valid and result are already updated here
        mac_en = 0;
        if (mac_out == 54) begin
            $display("  PASS: MAC result = %0d", mac_out);
            pass_count++;
        end else begin
            $display("  FAIL: MAC result = %0d (expected 54)", mac_out);
            fail_count++;
        end

        // ------------------------------------------------------------------
        // TEST 2: CONV — 4x4 input (values 1..16), 3x3 all-ones kernel
        //         Window [0][0]: sum(1,2,3,5,6,7,9,10,11) = 54
        //         Window [0][1]: sum(2,3,4,6,7,8,10,11,12) = 63
        //         Window [1][0]: sum(5,6,7,9,10,11,13,14,15) = 90
        //         Window [1][1]: sum(6,7,8,10,11,12,14,15,16) = 99
        // ------------------------------------------------------------------
        $display("\n=== TEST 2: Conv_hw ===");
        for (r = 0; r < IMG_H; r++)
            for (c = 0; c < IMG_W; c++)
                fm[0][r][c] = r * IMG_W + c + 1;

        for (r = 0; r < KS; r++)
            for (c = 0; c < KS; c++)
                wt[0][r][c] = 8'sd1;

        @(posedge clk); #1;
        conv_start = 1;
        @(posedge clk); #1;
        conv_start = 0;

        while (conv_ready !== 1'b1) begin @(posedge clk); #1; end
        if (conv_result[0][0] == 54 && conv_result[0][1] == 63 &&
            conv_result[1][0] == 90 && conv_result[1][1] == 99) begin
            $display("  PASS: Conv outputs = {%0d, %0d, %0d, %0d}",
                     conv_result[0][0], conv_result[0][1],
                     conv_result[1][0], conv_result[1][1]);
            pass_count++;
        end else begin
            $display("  FAIL: Conv outputs = {%0d, %0d, %0d, %0d} (expected {54,63,90,99})",
                     conv_result[0][0], conv_result[0][1],
                     conv_result[1][0], conv_result[1][1]);
            fail_count++;
        end

        // ------------------------------------------------------------------
        // TEST 3: POOL — 4x4 input (1..16), 2x2 AVG pool
        //         Window [0][0]: avg(1,2,5,6) = 14>>2 = 3
        //         Window [0][1]: avg(3,4,7,8) = 22>>2 = 5
        //         Window [1][0]: avg(9,10,13,14) = 46>>2 = 11
        //         Window [1][1]: avg(11,12,15,16) = 54>>2 = 13
        // ------------------------------------------------------------------
        $display("\n=== TEST 3: Pool (AVG 2x2) ===");
        pool_in[0][0] =  1; pool_in[0][1] =  2; pool_in[0][2] =  3; pool_in[0][3] =  4;
        pool_in[1][0] =  5; pool_in[1][1] =  6; pool_in[1][2] =  7; pool_in[1][3] =  8;
        pool_in[2][0] =  9; pool_in[2][1] = 10; pool_in[2][2] = 11; pool_in[2][3] = 12;
        pool_in[3][0] = 13; pool_in[3][1] = 14; pool_in[3][2] = 15; pool_in[3][3] = 16;

        @(posedge clk); #1;
        pool_start = 1;
        @(posedge clk); #1;
        pool_start = 0;

        while (pool_valid !== 1'b1) begin @(posedge clk); #1; end
        if (pool_out[0][0] == 3 && pool_out[0][1] == 5 &&
            pool_out[1][0] == 11 && pool_out[1][1] == 13) begin
            $display("  PASS: Pool outputs = {%0d, %0d, %0d, %0d}",
                     pool_out[0][0], pool_out[0][1],
                     pool_out[1][0], pool_out[1][1]);
            pass_count++;
        end else begin
            $display("  FAIL: Pool outputs = {%0d, %0d, %0d, %0d} (expected {3,5,11,13})",
                     pool_out[0][0], pool_out[0][1],
                     pool_out[1][0], pool_out[1][1]);
            fail_count++;
        end

        // ------------------------------------------------------------------
        // TEST 4: RELU — input [-3, -1, 0, 2, 5] -> expect [0, 0, 0, 2, 5]
        // ------------------------------------------------------------------
        $display("\n=== TEST 4: ReLU ===");
        relu_in[0] = -8'sd3;
        relu_in[1] = -8'sd1;
        relu_in[2] =  8'sd0;
        relu_in[3] =  8'sd2;
        relu_in[4] =  8'sd5;

        @(posedge clk); #1;
        relu_start = 1;
        @(posedge clk); #1;
        relu_start = 0;

        while (relu_valid !== 1'b1) begin @(posedge clk); #1; end
        if (relu_out[0] == 0 && relu_out[1] == 0 && relu_out[2] == 0 &&
            relu_out[3] == 2 && relu_out[4] == 5) begin
            $display("  PASS: ReLU outputs = {%0d, %0d, %0d, %0d, %0d}",
                     relu_out[0], relu_out[1], relu_out[2],
                     relu_out[3], relu_out[4]);
            pass_count++;
        end else begin
            $display("  FAIL: ReLU outputs = {%0d, %0d, %0d, %0d, %0d} (expected {0,0,0,2,5})",
                     relu_out[0], relu_out[1], relu_out[2],
                     relu_out[3], relu_out[4]);
            fail_count++;
        end

        // ------------------------------------------------------------------
        // TEST 5: DENSE — 3->2 FC layer
        //         in = [1, 2, 3]
        //         w[0] = [1, 1, 1], b[0] = 0  ->  out[0] = 1+2+3 = 6
        //         w[1] = [2, 0, 1], b[1] = 10 ->  out[1] = 2+0+3+10 = 15
        // ------------------------------------------------------------------
        $display("\n=== TEST 5: Dense (FC) ===");
        dense_in[0] = 8'sd1;
        dense_in[1] = 8'sd2;
        dense_in[2] = 8'sd3;

        dense_w[0][0] = 8'sd1; dense_w[0][1] = 8'sd1; dense_w[0][2] = 8'sd1;
        dense_w[1][0] = 8'sd2; dense_w[1][1] = 8'sd0; dense_w[1][2] = 8'sd1;

        dense_b[0] = '0;
        dense_b[1] = 10;

        @(posedge clk); #1;
        dense_start = 1;
        @(posedge clk); #1;
        dense_start = 0;

        while (dense_valid !== 1'b1) begin @(posedge clk); #1; end
        if (dense_out[0] == 6 && dense_out[1] == 15) begin
            $display("  PASS: Dense outputs = {%0d, %0d}", dense_out[0], dense_out[1]);
            pass_count++;
        end else begin
            $display("  FAIL: Dense outputs = {%0d, %0d} (expected {6, 15})",
                     dense_out[0], dense_out[1]);
            fail_count++;
        end

        // ------------------------------------------------------------------
        // TEST 6: FULL LENET FORWARD PASS
        //   All weights = 1, all biases = 0, image = all 1s
        //   Trace:
        //     C1:  each pixel = 1*1 * 25 terms = 25      -> ReLU/trunc = 25
        //     S2:  avg(25,25,25,25) = 25
        //     C3:  each pixel = 25*1 * 150 terms = 3750  -> saturates to 127
        //     S4:  avg(127,127,127,127) = 127
        //     FC5: 127*1 * 400 + 0 = 50800               -> saturates to 127
        //     FC6: 127*1 * 120 + 0 = 15240                -> saturates to 127
        //     FC7: 127*1 * 84  + 0 = 10668                -> raw output
        //   Expected: all 10 scores = 10668
        // ------------------------------------------------------------------
        $display("\n=== TEST 6: Full LeNet Forward Pass ===");
        $display("  Initializing weights and inputs...");

        // Image: all 1s
        for (int ri = 0; ri < 32; ri++)
            for (int ci = 0; ci < 32; ci++)
                lenet_image[ri][ci] = 8'sd1;

        // C1 weights: all 1s (6 filters x 1ch x 5x5)
        for (int f = 0; f < 6; f++)
            for (int ch = 0; ch < 1; ch++)
                for (int kr = 0; kr < 5; kr++)
                    for (int kc = 0; kc < 5; kc++)
                        lenet_c1_w[f][ch][kr][kc] = 8'sd1;

        // C3 weights: all 1s (16 filters x 6ch x 5x5)
        for (int f = 0; f < 16; f++)
            for (int ch = 0; ch < 6; ch++)
                for (int kr = 0; kr < 5; kr++)
                    for (int kc = 0; kc < 5; kc++)
                        lenet_c3_w[f][ch][kr][kc] = 8'sd1;

        // FC5 weights: all 1s (120 x 400)
        for (int j = 0; j < 120; j++)
            for (int i = 0; i < 400; i++)
                lenet_fc5_w[j][i] = 8'sd1;

        // FC5 bias: all 0
        for (int j = 0; j < 120; j++)
            lenet_fc5_b[j] = '0;

        // FC6 weights: all 1s (84 x 120)
        for (int j = 0; j < 84; j++)
            for (int i = 0; i < 120; i++)
                lenet_fc6_w[j][i] = 8'sd1;

        // FC6 bias: all 0
        for (int j = 0; j < 84; j++)
            lenet_fc6_b[j] = '0;

        // FC7 weights: all 1s (10 x 84)
        for (int j = 0; j < 10; j++)
            for (int i = 0; i < 84; i++)
                lenet_fc7_w[j][i] = 8'sd1;

        // FC7 bias: all 0
        for (int j = 0; j < 10; j++)
            lenet_fc7_b[j] = '0;

        $display("  Starting forward pass...");
        @(posedge clk); #1;
        lenet_start = 1;
        @(posedge clk); #1;
        lenet_start = 0;

        // Wait for completion with timeout
        begin
            integer timeout_cnt;
            timeout_cnt = 0;
            while (lenet_done !== 1'b1 && timeout_cnt < 500000) begin
                @(posedge clk); #1;
                timeout_cnt = timeout_cnt + 1;
            end
            if (lenet_done !== 1'b1) begin
                $display("  TIMEOUT: LeNet forward pass did not complete!");
                fail_count++;
            end
        end

        if (lenet_done === 1'b1) begin
            $display("  Forward pass completed successfully!");
            $display("  Output scores:");
            for (int s = 0; s < 10; s++)
                $display("    scores[%0d] = %0d", s, lenet_scores[s]);

            // Verify all scores equal expected value (10668)
            all_correct = 1;
            for (int s = 0; s < 10; s++)
                if (lenet_scores[s] !== 10668) all_correct = 0;

            if (all_correct) begin
                $display("  PASS: All scores = 10668 (matches expected)");
                pass_count++;
            end else begin
                $display("  FAIL: Expected all scores = 10668");
                fail_count++;
            end
        end

        // ------------------------------------------------------------------
        // Summary
        // ------------------------------------------------------------------
        $display("\n========================================");
        $display("  RESULTS: %0d passed, %0d failed", pass_count, fail_count);
        $display("========================================\n");
        $finish;
    end

    // Waveform dump (works on EDA Playground with most simulators)
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, tb_lenet_modules);
    end

endmodule
