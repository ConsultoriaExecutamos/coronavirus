<script>
import { Line } from "vue-chartjs";

export default {
  extends: Line,
  props: ["chartdata", "options", "gradientColors"],
  data() {
    return {
      gradient: null
    };
  },
  mounted() {
    const chartData = this.chartdata;
    const chartOptions = this.options;

    this.gradient = this.$refs.canvas
      .getContext("2d")
      .createLinearGradient(0, 0, 0, 450);
    if (this.gradientColors) {
      for (let idx = 0; idx < this.gradientColors.length; idx++) {
        this.gradient.addColorStop(idx, this.gradientColors[idx]);
      }
      chartData.datasets[0].backgroundColor = this.gradient;
    }

    this.renderChart(chartData, chartOptions);
  }
};
</script>

<style></style>
