<template>
  <div class="home">
    <section class="section">
      <div id="header" class="container">
        <div class="columns">
          <div class="column is-2">
            <img src="@/assets/corona.svg" alt="" srcset="" />
          </div>
          <div class="column is-4 has-text-left">
            <h1>COVID-19</h1>
            <h4>Análise</h4>
          </div>
        </div>
      </div>
      <hr />
      <div id="chart-container" class="container">
        <div class="columns">
          <div class="column is-half">
            <div class="box">
              <line-chart
                :chartdata="chartData"
                :options="chartOptions"
                :style="chartStyles"
              >
              </line-chart>
            </div>
          </div>
          <div class="column is-half">
            <h1>
              <span class="category"
                ><span class="underline">Previsão</span>:
              </span>
              <span class="number-info">200.000</span> casos totais até 26
              abril.
            </h1>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script>
import LineChart from "@/components/LineChart";

export default {
  name: "Home",
  components: { LineChart },
  data() {
    return {
      chartData: {
        labels: [
          "Feb 15",
          "Feb 16",
          "Feb 17",
          "Feb 18",
          "Feb 19",
          "Feb 20",
          "Feb 21",
          "Feb 22",
          "Feb 23",
          "Feb 24",
          "Feb 25",
          "Feb 26",
          "Feb 27",
          "Feb 28",
          "Feb 29",
          "Mar 1",
          "Mar 2",
          "Mar 3",
          "Mar 4",
          "Mar 5",
          "Mar 6",
          "Mar 7",
          "Mar 8",
          "Mar 9",
          "Mar 10",
          "Mar 11",
          "Mar 12",
          "Mar 13",
          "Mar 14",
          "Mar 15",
          "Mar 16",
          "Mar 17",
          "Mar 18",
          "Mar 19",
          "Mar 20",
          "Mar 21",
          "Mar 22",
          "Mar 23",
          "Mar 24",
          "Mar 25",
          "Mar 26"
        ],
        datasets: [
          {
            label: "Total de Casos",
            backgroundColor: "rgba(0,0,0,0)",
            borderColor: "rgb(255, 99, 132)",
            data: [
              15,
              15,
              15,
              15,
              15,
              15,
              35,
              35,
              35,
              53,
              57,
              60,
              60,
              63,
              68,
              75,
              100,
              124,
              158,
              221,
              319,
              435,
              541,
              704,
              994,
              1301,
              1630,
              2183,
              2770,
              3613,
              4596,
              6344,
              9197,
              13779,
              19367,
              24192,
              33592,
              43781,
              54856,
              68211,
              85435
            ]
          }
        ]
      },
      chartOptions: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          xAxes: [
            {
              gridLines: {
                display: false
              }
            }
          ]
        }
      }
    };
  },
  computed: {
    chartStyles() {
      return {
        "max-height": "200rem"
      };
    }
  }
};
</script>

<style lang="scss">
@import "@/styles/theme.scss";

body {
  min-height: 100vh;
  background-image: linear-gradient(
    to bottom,
    #1e1e2e,
    #1e1e2c,
    #1e1e29,
    #1e1e27,
    #1e1e24
  );
  background-color: #2f3542;
}

#header {
  .columns {
    justify-content: center;
    align-items: center;
  }
  h1 {
    font-size: 4.5rem;
    font-weight: 700;
  }

  h4 {
    font-size: 2.5rem;
    line-height: 2rem;
    font-weight: 600;
  }

  img {
    display: block;
    width: 100%;
  }
}

#chart-container {
  h1 {
    font-weight: normal;
    font-size: 2rem;

    .category {
      font-weight: 600;
    }

    .underline {
      text-decoration: underline;
      text-decoration-color: $primary;
    }

    .number-info {
      font-weight: 800;
      font-size: 2.5rem;
    }
  }
}
</style>
