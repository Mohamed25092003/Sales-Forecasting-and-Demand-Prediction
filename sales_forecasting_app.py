
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import re


file_path = './dataset/sales_Depi.csv'
df = pd.read_csv(file_path)
df.drop_duplicates(inplace=True)

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])
df["Day"]=df["Order Date"].dt.day
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["DayOfWeek"] = df["Order Date"].dt.dayofweek
df["IsWeekend"] = df["DayOfWeek"] >= 5
df["IsWeekend"]=df["IsWeekend"].astype(int)
df = df.sort_values(by='Order Date')
df["Quarter"] = df["Order Date"].dt.quarter
# Convert timedelta to number of days
df["Shipping Delay"]=df["Ship Date"]-df["Order Date"]
df["Shipping Delay"] = df["Shipping Delay"].dt.days

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df["Season"] = df["Order Date"].dt.month.apply(get_season)


def target_encoding(df_enc, features, target_col="Units Sold"):
    encoder = TargetEncoder()
    encoded = encoder.fit_transform(df_enc[features], df_enc[target_col])
    
    # Rename encoded columns to keep original names + "_encoded"
    encoded.columns = [f"{col}_encoded" for col in features]
    
    # Concatenate with original dataframe
    df_enc = df_enc.copy()
    df_enc[encoded.columns] = encoded

    return df_enc
def one_hot_encoding(df_enc, features):
    for i,col in enumerate(features):
        dummies = pd.get_dummies(df_enc[col], drop_first=True)
        # Rename each dummy column to value_Encoded format
        dummies.columns = [f"{features[i]}_{val}" for val in dummies.columns]
        df_enc = pd.concat([df_enc, dummies], axis=1)
    return df_enc

def label_encoding(df_enc, features):
    for col in features:
        encoder = LabelEncoder()
        clean_col_name = col.strip().replace(" ", "_")
        df_enc[f"{clean_col_name}_Encoded"] = encoder.fit_transform(df_enc[col])
    return df_enc




def show_plots(plot_type):
    if plot_type=="Overall Yearly Sales Trend":
        sales_trend = df.groupby('Year')['Total Revenue'].sum().reset_index()
        sales_trend["Year"] = sales_trend["Year"].astype(str)
        fig1 = px.line(
            sales_trend,
            x='Year',
            y='Total Revenue',
            title='Overall Yearly Sales Trend',
            markers=True,
            labels={'Year': 'Year', 'Total Revenue': 'Total Sales'},
        )
        fig1.update_traces(line=dict(color='blue', width=2))
        fig1.update_layout(
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            template='plotly_white',
            xaxis=dict(type="category"),
            yaxis=dict(range=[0, max(sales_trend["Total Revenue"]) * 1.1])  # Dynamic range
        )
        st.plotly_chart(fig1)

    elif plot_type=="Overall Sales Trend Over Time":
        sales_trend = df.groupby('Order Date')['Total Revenue'].sum().reset_index()
        sales_trend['Smoothed_Sales'] = sales_trend['Total Revenue'].rolling(window=7).mean()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sales_trend['Order Date'],
            y=sales_trend['Smoothed_Sales'],
            mode='lines',
            name='7-Day Moving Avg',
            line=dict(color='blue', width=2)
        ))
        fig2.update_layout(
            title='Overall Sales Trend Over Time',
            xaxis_title='Order Date',
            yaxis_title='Total Sales',
            template='plotly_white',
            width=1000,
            height=500,
            yaxis=dict(range=[0, max(sales_trend['Smoothed_Sales'].dropna()) * 1.1])  # Dynamic range
        )
        st.plotly_chart(fig2)

    elif plot_type == "Monthly Revenue and Profit Trends":
        monthly_revenue = df.groupby(['Year', 'Month']).agg({
            'Total Revenue': 'sum',
            'Total Profit': 'sum'
        }).reset_index()
    
        monthly_revenue['date'] = pd.to_datetime(
            monthly_revenue['Year'].astype(str) + '-' +
            monthly_revenue['Month'].astype(str).str.zfill(2) + '-01'
        )
        monthly_revenue = monthly_revenue.sort_values('date')
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=monthly_revenue['date'],
            y=monthly_revenue['Total Revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#3498db', width=2)
        ))
        fig3.add_trace(go.Scatter(
            x=monthly_revenue['date'],
            y=monthly_revenue['Total Profit'],
            mode='lines+markers',
            name='Profit',
            line=dict(color='#2ecc71', width=2)
        ))
        fig3.update_layout(
            title='Monthly Revenue and Profit Trends',
            xaxis_title='Date',
            yaxis_title='Amount ($)',
            template='plotly_white',
            legend=dict(font=dict(size=12)),
            xaxis=dict(tickangle=45),
            width=1000,
            height=550,
            yaxis=dict(range=[0, max(max(monthly_revenue["Total Revenue"]), max(monthly_revenue["Total Profit"])) * 1.1])
        )
        st.plotly_chart(fig3)
    elif plot_type=="Monthly Sales Over Years by Order Priority":
        #  Get unique priorities
            priorities = df['Order Priority'].unique()
            num_priorities = len(priorities)
            num_cols = 2
            num_rows = (num_priorities + 1) // 2         

            # Create subplot grid
            fig4 = make_subplots(
                rows=num_rows,
                cols=num_cols,
                subplot_titles=[f"{p} Priority - Monthly Sales" for p in priorities],
                shared_yaxes=True
            )

            # Loop through each priority and create a trace
            for idx, priority in enumerate(priorities):
                row = idx // num_cols + 1
                col = idx % num_cols + 1

                priority_data = df[df['Order Priority'] == priority]
                monthly_sales = priority_data.groupby(['Year', 'Month'])['Total Revenue'].sum().reset_index()
                monthly_sales['YearMonth'] = pd.to_datetime(
                    monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str).str.zfill(2) + '-01'
                )

                fig4.add_trace(
                    go.Scatter(
                        x=monthly_sales['YearMonth'],
                        y=monthly_sales['Total Revenue'],
                        mode='lines+markers',
                        name=str(priority),
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )

            # Final layout tweaks
            fig4.update_layout(
                height=300 * num_rows,
                width=1000,
                title_text="Monthly Sales Over Years by Order Priority",
                template='plotly_white'
            )

            fig4.update_xaxes(title_text="Date")
            fig4.update_yaxes(title_text="Total Revenue")
            st.plotly_chart(fig4)

    elif plot_type == "Monthly Sales Across Years by Region":
         # Get unique regions
        regions = df['Region'].unique()
        num_regions = len(regions)
        num_cols = 2
        num_rows = (num_regions + 1) // 2

        # Create subplot layout     
        fig5 = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=[f"{region} Region - Monthly Sales" for region in regions],
            shared_yaxes=True
        )

        # Add traces per region
        for idx, region in enumerate(regions):
            row = idx // num_cols + 1
            col = idx % num_cols + 1

            region_data = df[df['Region'] == region]
            monthly_sales = region_data.groupby(['Year', 'Month'])['Total Revenue'].sum().reset_index()
            monthly_sales['YearMonth'] = pd.to_datetime(
                monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str).str.zfill(2) + '-01'
            )

            fig5.add_trace(
                go.Scatter(
                    x=monthly_sales['YearMonth'],
                    y=monthly_sales['Total Revenue'],
                    mode='lines+markers',
                    name=region,
                    showlegend=False
                ),
                row=row,
                col=col
            )

        # Layout adjustments
        fig5.update_layout(
            height=300 * num_rows,
            width=1150,
            title_text="Monthly Sales Across Years by Region",
            template='plotly_white'
        )

        fig5.update_xaxes(title_text="Date")
        fig5.update_yaxes(title_text="Total Revenue")
        st.plotly_chart(fig5)

    elif plot_type=="Sales Trend by Item Type":
        # Prepare figure
        fig6 = go.Figure()   ####6

        item_types = df['Item Type'].unique()
        buttons = []

        # Add one trace per item type
        for i, item_type in enumerate(item_types):
            subset = df[df['Item Type'] == item_type]
            daily_sales = subset.groupby('Order Date')['Total Revenue'].sum().reset_index()

            fig6.add_trace(go.Scatter(
                x=daily_sales['Order Date'],
                y=daily_sales['Total Revenue'],
                mode='lines+markers',
                name=item_type,
                visible=(i == 0)
            ))

            # Dropdown button for each item type
            visibility = [False] * len(item_types)
            visibility[i] = True
            buttons.append(dict(label=item_type,
                                method="update",
                                args=[{"visible": visibility},
                                    {"title": f"<b>Sales Trend for Item Type:</b> {item_type}"}]))

        # Layout with static and dynamic titles
        fig6.update_layout(
            title="<b>Sales Trend by Item Type</b><br><span style='font-size:14px'>Sales Trend for Item Type: {}</span>".format(item_types[0]),
            xaxis_title="Order Date",
            yaxis_title="Total Sales",
            template="plotly_white",
            width=1000,
            height=550,
            updatemenus=[{
                "active": 0,
                "buttons": buttons,
                "x": 1.05,
                "y": 1.2,
                "xanchor": "right"
            }]
        )
        st.plotly_chart(fig6)
    elif plot_type == "Proportion of Different Item Types":
        # Prepare data
        membership_counts = df['Item Type'].value_counts()  #####7

        # Create Plotly pie chart
        fig7 = go.Figure(data=[go.Pie(
            labels=membership_counts.index,
            values=membership_counts.values,
            hole=0,  # set to >0 (like 0.4) if you want a donut chart
            hoverinfo='label+percent+value',
            textinfo='percent',
            textfont_size=14
        )])

        # Layout
        fig7.update_layout(
            title_text='Proportion of Different Item Types',
            height=600,
            width=600,
            showlegend=True
        )
        st.plotly_chart(fig7)
    elif plot_type=="Number of Purchases by Sales Channel":
        # Prepare data
        referral_purchases = df['Sales Channel'].value_counts().reset_index()
        referral_purchases.columns = ['Sales Channel', 'Count']

        # Create Plotly bar chart
        fig8 = go.Figure(data=[go.Bar(
            x=referral_purchases['Sales Channel'],
            y=referral_purchases['Count'],                                           #####8
            marker=dict(color='#3498db'),  # You can customize the color here
        )])

        # Layout customization
        fig8.update_layout(
            title='Number of Purchases by Sales Channel',
            xaxis_title='Sales Channel',
            yaxis_title='Number of Purchases',
            xaxis_tickangle=-45,
            template='plotly_white',
            height=500,
            width=800
        )
        st.plotly_chart(fig8)
    elif plot_type=="Monthly Revenue by Year":
        # Monthly sales by year
        monthly_pivot = df.pivot_table(
            index='Month', 
            columns='Year', 
            values='Total Revenue', 
            aggfunc='sum'
        ).reset_index()

        # Convert wide format to long format for plotly           #####9
        monthly_long = pd.melt(
            monthly_pivot, 
            id_vars=['Month'], 
            var_name='Year', 
            value_name='revenue'
        )

        # Add month names
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
            7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        monthly_long['month_name'] = monthly_long['Month'].map(month_names)

        # Create interactive heatmap
        fig9 = px.density_heatmap(
            monthly_long,
            x='Year',
            y='month_name',
            z='revenue',
            color_continuous_scale='Blues',
            title='Monthly Revenue by Year',
            labels={'revenue': 'Revenue ($)', 'Year': 'Year', 'month_name': 'Month'},
            text_auto=True  # Automatically determine format
        )

        # Update layout for better display of large numbers
        fig9.update_layout(
            coloraxis_colorbar=dict(
                title="Revenue ($)",
                tickprefix="$",
                tickformat=",.0f"  # Comma separated with no decimal places
            ),
            height=700,
            yaxis=dict(
                categoryorder='array',
                categoryarray=list(month_names.values())  # Ensure correct month order
            )
        )

        # Format text on cells to be in thousands with dollar sign
        fig9.update_traces(
            texttemplate='$%{z:,.0f}',
            textfont={"size": 12}
        )
        st.plotly_chart(fig9)

    elif plot_type=="Top 5 Countries by Revenue":
        # Prepare data
        top_countries = df.groupby('Country')['Total Revenue'].sum().nlargest(5)

        # Create Plotly horizontal bar chart
        fig10 = go.Figure(data=[go.Bar(
            y=top_countries.index,                           ######10
            x=top_countries.values,
            orientation='h',  # Horizontal bars
            marker=dict(color='purple')
        )])

        # Layout customization
        fig10.update_layout(
            title='Top 5 Countries by Revenue',
            xaxis_title='Total Revenue ($)',
            yaxis_title='Country',
            template='plotly_white',
            height=500,
            width=800
        )
        st.plotly_chart(fig10)

    elif plot_type=="Revenue and Profit by Order Priority":
        # Prepare data
        priority_data = df.groupby('Order Priority').agg({
            'Total Revenue': 'sum',
            'Total Profit': 'sum'
        }).reset_index()                       #####11

        # Create Plotly grouped bar chart
        fig11 = go.Figure(data=[
            go.Bar(
                name='Revenue',
                x=priority_data['Order Priority'],
                y=priority_data['Total Revenue'],
                marker_color='steelblue'
            ),
            go.Bar(
                name='Profit',
                x=priority_data['Order Priority'],
                y=priority_data['Total Profit'],
                marker_color='seagreen'
            )
        ])

        # Layout settings
        fig11.update_layout(
            title='Revenue and Profit by Order Priority',
            xaxis_title='Order Priority',
            yaxis_title='Amount ($)',
            barmode='group',  # side-by-side bars
            template='plotly_white',
            width=800,
            height=500
        )
        st.plotly_chart(fig11)

    elif plot_type=="Revenue by Product Category":
        # Product category analysis
        product_data = df.groupby('Item Type')['Total Revenue'].sum().reset_index()

        # Create dynamic pie chart with Plotly Express
        fig12 = px.pie(                                                      ######12
            product_data, 
            values='Total Revenue', 
            names='Item Type',
            title='Revenue by Product Category',
            color_discrete_sequence=px.colors.qualitative.Bold,
            # Add hover information
            hover_data=['Total Revenue'],
            labels={'Total Revenue': 'Revenue ($)'}
        )

        # Improve layout for better readability
        fig12.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hole=0.3,
            pull=[0.05 if product_data['Total Revenue'].iloc[i] == product_data['Total Revenue'].max() else 0 
                for i in range(len(product_data))]  # Pull out the largest slice
        )

        # Update layout
        fig12.update_layout(
            title_font_size=24,
            legend_title="Product Categories",
            font=dict(size=14)
        )
        st.plotly_chart(fig12)
    
    elif plot_type=="Profit by Region":
        # Prepare data
        region_data = df.groupby('Region')['Total Profit'].sum().sort_values(ascending=False)

        # Create Plotly bar chart
        fig13 = go.Figure(data=go.Bar(
            x=region_data.index,
            y=region_data.values,
            marker_color='green',
            name='Total Profit'
        ))                                            #####13

        # Layout settings
        fig13.update_layout(
            title='Profit by Region',
            xaxis_title='Region',
            yaxis_title='Total Profit ($)',
            template='plotly_white',
            width=800,
            height=500
        )
        st.plotly_chart(fig13)

    elif plot_type=="Profit Margin by Product Type":
        # Calculate profit margin by product type (fixing the deprecation warning)
        product_data = df.groupby('Item Type', as_index=False).agg({
            'Total Profit': 'sum',
            'Total Revenue': 'sum'
        })                                      

        # Calculate profit margin percentage
        product_data['profit_margin'] = 100 * product_data['Total Profit'] / product_data['Total Revenue']

        # Sort by profit margin
        product_data = product_data.sort_values('profit_margin', ascending=False)

        # Create a Plotly bar chart for better interactivity
        fig14 = px.bar(
            product_data,
            x='Item Type',
            y='profit_margin',
            color='profit_margin',
            color_continuous_scale='teal',
            text=product_data['profit_margin'].round(1).astype(str) + '%',
            title='Profit Margin by Product Type',
            labels={'profit_margin': 'Profit Margin (%)', 'item_type': 'Product Type'},
            height=600
        )

        # Enhance layout
        fig14.update_layout(                                                  #####14
            xaxis_title='Product Type',
            yaxis_title='Profit Margin (%)',
            yaxis=dict(
                ticksuffix='%',
                gridcolor='lightgray',
                gridwidth=0.5,
            ),
            coloraxis_showscale=False,  # Hide color scale
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
            )
        )

        # Enhance hover information
        fig14.update_traces(
            hovertemplate='<b>%{x}</b><br>Profit Margin: %{y:.1f}%<br>Revenue: $%{customdata[0]:,.0f}<br>Profit: $%{customdata[1]:,.0f}',
            customdata=product_data[['Total Revenue', 'Total Profit']],
            textposition='outside'
        )
        st.plotly_chart(fig14)

    elif plot_type=="Unit Price vs Units Sold":
        fig15 = px.scatter(
        df, 
        x='Unit Price', 
        y='Units Sold',
        color='Item Type',         
        title='Unit Price vs Units Sold',
        opacity=0.7,
        height=500,
        width=700
    )

        # Minimal layout customization
        fig15.update_layout(
            xaxis_title="Unit Price ($)",
            yaxis_title="Units Sold",
            legend_title="Product Type"
        )
        st.plotly_chart(fig15)

    elif plot_type=="Units Sold by Day of Month":
        # Prepare data
        daily_units = df.groupby('Day')['Units Sold'].sum()
        # Create Plotly line chart
        fig15 = go.Figure(data=go.Scatter(
            x=daily_units.index,
            y=daily_units.values,
            mode='lines+markers',
            line=dict(color='royalblue'),
            marker=dict(size=6),
            name='Units Sold'
        ))

        # Layout settings
        fig15.update_layout(
            title='Units Sold by Day of Month',
            xaxis_title='Day of Month',
            yaxis_title='Units Sold',
            template='plotly_white',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),  # Show ticks for all days 1–31
            width=800,
            height=500
            )
        st.plotly_chart(fig15)
    elif plot_type=="Distribution of Units Sold":
        fig16 = px.histogram(df, x='Units Sold', nbins=1000)
        fig16.update_layout(
            title={
                'text': 'Distribution of Units Sold',
                'x': 0.5,
                'xanchor': 'center'
            }
)
        st.plotly_chart(fig16)
        
    elif plot_type=="Monthly Demand with Rolling Average and Seasonal Shading":
        df_temp = df.copy()

        # Ensure 'Order Date' is datetime and set it as the index
        df_temp['Order Date'] = pd.to_datetime(df_temp['Order Date'])
        df_temp.set_index('Order Date', inplace=True)

        # Now resample monthly
        monthly_demand = df_temp['Units Sold'].resample('M').sum()
        rolling_avg = monthly_demand.rolling(window=3).mean()

        # Create base figure
        fig17 = go.Figure()

        # Monthly Demand line
        fig17.add_trace(go.Scatter(
            x=monthly_demand.index,
            y=monthly_demand.values,
            mode='lines+markers',
            name='Monthly Demand',
            line=dict(color='blue'),
            marker=dict(symbol='circle')
        ))

        # Rolling average line
        fig17.add_trace(go.Scatter(
            x=rolling_avg.index,
            y=rolling_avg.values,
            mode='lines',
            name='Rolling Avg (3-month)',
            line=dict(color='orange', width=3)
        ))

        # Add seasonal shading: Summer (June to August) and Winter (Dec to Feb)
        years = monthly_demand.index.year.unique()
        for year in years:
            # Summer shading
            fig17.add_vrect(
                x0=f'{year}-06-01', x1=f'{year}-08-31',
                fillcolor='skyblue', opacity=0.1, line_width=0
            )
            # Winter shading (Dec to Feb next year)
            fig17.add_vrect(
                x0=f'{year}-12-01', x1=f'{year+1}-02-28',
                fillcolor='lightgray', opacity=0.15, line_width=0
            )

        # Annotate minimum demand (February)
        min_month = monthly_demand.idxmin()
        min_value = monthly_demand.min()

        fig17.add_annotation(
            x=min_month,
            y=min_value,
            text='February has the lowest demand<br>across the years',
            showarrow=True,
            arrowhead=2,
            arrowcolor='red',
            font=dict(color='red', size=12),
            yshift=60
        )

        # Final layout
        fig17.update_layout(
            title={'text':'Monthly Demand with Rolling Average and Seasonal Shading', 'x':0.5, 'xanchor':'center'},
            xaxis_title='Month',
            yaxis_title='Units Sold',
            template='plotly_white',
            xaxis=dict(tickformat='%b %Y'),
            margin=dict(t=50, l=50, r=30, b=80),
            showlegend=True,
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig17)
    elif plot_type=="Profit vs Units Sold by Item Type":
# Scatter plot with Plotly
        fig18 = px.scatter(df, x="Units Sold", y="Total Profit", color="Item Type", color_discrete_sequence=px.colors.qualitative.Set3)

        fig18.update_layout(title={'text':'Profit vs. Units Sold by Item Type', 'x':0.5, 'xanchor':'center'})
        st.plotly_chart(fig18)        
    elif plot_type=="Revenue and Profit per Item Type":

        grouped = df.groupby('Item Type')[['Total Profit', 'Total Revenue','Total Cost','Units Sold']].sum()
        grouped = grouped.sort_values('Total Revenue', ascending=False)

        # Calculate Revenue and Profit per Unit
        grouped['Revenue per Unit'] = grouped['Total Revenue'] / grouped['Units Sold']
        grouped['Profit per Unit'] = grouped['Total Profit'] / grouped['Units Sold']

        # Prepare data for Plotly
        plot_data = grouped[['Profit per Unit','Revenue per Unit']].reset_index().melt(
            id_vars='Item Type',
            value_vars=['Profit per Unit','Revenue per Unit'],
            var_name='Metric',
            value_name='Amount'
        )

        fig19 = px.bar(
            plot_data,
            x='Amount',
            y='Item Type',
            color='Metric',
            orientation='h',
            barmode='group',
            labels={'Amount': 'Amount ($)', 'Item Type': 'Item Type'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        fig19.update_layout(title={
                'text': 'Revenue and Profit per Unit by Item Type',
                'x': 0.5,
                'xanchor': 'center'},
            template="plotly_white",
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig19)        
    elif plot_type=="Total Profit by Region":

        demand_by_region = df.groupby('Region')['Units Sold'].sum().reset_index()

        # Create Plotly pie chart with labels on the slices
        fig20 = px.pie(
            demand_by_region,
            names='Region',
            values='Units Sold',
            title='Total Profit by Region',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0
        )

        # Show percentage labels on the chart slices
        fig20.update_traces(textinfo='label+percent', textposition='outside', showlegend=False)
        fig20.update_layout(title={'text':'Total Profit by Region', 'x':0.5, 'xanchor':'center'})  # Center the title
        st.plotly_chart(fig20)        

    elif plot_type=="Units Sold by Item Type":
        item_type_units = df.groupby('Item Type')['Units Sold'].sum().sort_values(ascending=False).reset_index()

        fig21 = px.bar(
            item_type_units,
            x='Units Sold',
            y='Item Type',
            orientation='h',
            color='Item Type',
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        min_limit = item_type_units['Units Sold'].min() * 0.9
        max_limit = item_type_units['Units Sold'].max() * 1.01
        fig21.update_layout(title={
                'text': 'Units Sold by Item Type',
                'x': 0.5,
                'xanchor': 'center'},
            xaxis_range=[min_limit, max_limit],
            showlegend=False,
            template='plotly_white'
        )

        
        st.plotly_chart(fig21)        
    elif plot_type=="Units Sold per Item Type":
        fig22 = px.box(df, x='Item Type', y='Units Sold', color='Item Type', color_discrete_sequence=px.colors.qualitative.Set2) #Creating box plots for each item type
        fig22.update_layout(title={
                'text': 'Units Sold per Item Type',
                'x': 0.5,
                'xanchor': 'center'
            },showlegend=False) #Setting the layout
        st.plotly_chart(fig22)        

    elif plot_type=="Average Units Sold per Month (Seasonality)":
        df['Month'] = df['Order Date'].dt.month

        # Group by Month and Item Type
        monthly_sales = df.groupby(["Month", "Item Type"])["Units Sold"].mean().reset_index()

        fig23 = px.line(
            monthly_sales,
            x="Month",
            y="Units Sold",
            color="Item Type",
            markers=True,
            labels={"Units Sold": "Average Units Sold", "Month": "Month"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig23.update_layout(title={
                'text': 'Average Units Sold by Month (Seasonality)',
                'x': 0.5,
                'xanchor': 'center'},
            xaxis=dict(dtick=1),
            template="plotly_white"
        )

        st.plotly_chart(fig23)        
    elif plot_type=="Sales Channel Distribution":
        fig24 = px.pie(
            df,
            names='Sales Channel',
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        fig24.update_layout(title={
                'text': 'Sales Channel Distribution',
                'x': 0.5,
                'xanchor': 'center'
            },showlegend=False)

        # Show category labels directly on the pie slices
        fig24.update_traces(textinfo='label+percent', showlegend=False)
        st.plotly_chart(fig24)        

def show_models(item, df_enc):
    st.title("📊 Sales Prediction from Uploaded Data")

    uploaded_file = st.file_uploader("📤 Upload your CSV file for prediction", type=["csv"])
    if uploaded_file is None:
        return

    new_data = pd.read_csv(uploaded_file)

    if item == "all":
        try:
            with open('./models/model_all_UNITS.pkl', 'rb') as f:
                units_data = pickle.load(f)
            units_model = units_data['model']
            units_features = units_data['feature_names']


            with open('./models/model_all_revenue.pkl', 'rb') as f:
                revenue_data = pickle.load(f)
            revenue_model = revenue_data['model']
            revenue_features = revenue_data['feature_names']
        except FileNotFoundError:
            st.error("❌ One or both of the 'all' models are missing.")
            return

        new_data = feature_handling(new_data, df_enc, item)

        # Predict both outputs
        new_data["Predicted Units Sold"] = units_model.predict(new_data[units_features])
        new_data["Predicted Total Revenue"] = revenue_model.predict(new_data[revenue_features])

        st.success("✅ Prediction Complete (Units Sold + Total Revenue)!")
        st.dataframe(new_data[["Predicted Units Sold","Predicted Total Revenue"]])

        csv = new_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "predicted_all.csv", "text/csv")
        return

    # Map of item types to their model filenames
    item_model_map = {
        "Baby Food": "xgb_model_baby_food.pkl",
        "Clothes": "xgb_model_clothes.pkl",
        "Cosmetics": "xgb_model_cosmetics.pkl",
        "Cereal": "xgb_model_with_cereal_features.pkl",
        "Fruits": "model_fruit.pkl",
        "Meat": "XGBoost_Meat_Model.pkl",
        "Beverages": "XGBoost_Beverages_Model.pkl",
        "Office Supplies": "model_office.pkl",
        "Personal Care": "model_personalcare.pkl",
        "Vegetables": "model_vege.pkl"
    }

    model_file = item_model_map.get(item)
    if not model_file:
        st.error("🚫 Model not found for the selected item.")
        return

    try:
        with open(f'./models/{model_file}', 'rb') as f:
            data = pickle.load(f)
        model = data['model']
        feature_names = data['feature_names']
    except FileNotFoundError:
        st.error(f"❌ Model file for '{item}' not found.")
        return

    new_data = feature_handling(new_data, df_enc, item)

    new_data["Predicted Units Sold"] = model.predict(new_data[feature_names])

    st.success("✅ Prediction Complete!")
    st.dataframe(new_data[["Predicted Units Sold"]])

    csv = new_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results", csv, "predicted_units_sold.csv", "text/csv")



def feature_handling(df_new,df_ref,item):
    df_new["Order Date"] = pd.to_datetime(df_new["Order Date"])
    df_new["Ship Date"] = pd.to_datetime(df_new["Ship Date"])
    df_new["Day"]=df_new["Order Date"].dt.day
    df_new["Year"] = df_new["Order Date"].dt.year
    df_new["Month"] = df_new["Order Date"].dt.month
    df_new["DayOfWeek"] = df_new["Order Date"].dt.dayofweek
    df_new["Week"] = df_new["Order Date"].dt.isocalendar().week
    df_new["IsWeekend"] = df_new["DayOfWeek"] >= 5
    df_new["IsWeekend"]=df_new["IsWeekend"].astype(int)
    df_new = df_new.sort_values(by='Order Date')
    df_new["Quarter"] = df_new["Order Date"].dt.quarter
    # Convert timedelta to number of days
    df_new["Shipping Delay"]=df_new["Ship Date"]-df_new["Order Date"]
    df_new["Shipping Delay"] = df_new["Shipping Delay"].dt.days
    df_new["Season"] = df_new["Order Date"].dt.month.apply(get_season)
    df_new = df_new.set_index('Order Date')  
    monthly_df_new = df_new['Units Sold'].resample('M').sum().to_frame()
    monthly_df_new_rev = df_new['Total Revenue'].resample('M').sum().to_frame()

    df_new['Profit Margin'] = df_new['Total Profit'] / df_new['Total Revenue']
    df_new['Markup'] = (df_new['Unit Price'] - df_new['Unit Cost']) / df_new['Unit Cost']
    # Lag features
    df_new["Lag_1D"] = df_new["Units Sold"].shift(1)
    df_new["Lag_7D"] = df_new["Units Sold"].shift(7)
    df_new["Lag_14D"] = df_new["Units Sold"].shift(14)
    df_new["Lag_21D"] = df_new["Units Sold"].shift(21)
    df_new["Lag_1"] = monthly_df_new["Units Sold"].shift(1)
    df_new["Lag_3"] = monthly_df_new["Units Sold"].shift(3)
    df_new["Lag_6"] = monthly_df_new["Units Sold"].shift(6)
    df_new["Lag_12"]= monthly_df_new["Units Sold"].shift(12)

    df_new['Lag_1D_rev'] = df_new['Total Revenue'].shift(1)
    df_new['Lag_7D_rev'] = df_new['Total Revenue'].shift(7)
    # Rolling averages
    df_new["Rolling_7D"] = df_new["Units Sold"].rolling(window=7).mean()
    df_new["Rolling_14D"] = df_new["Units Sold"].rolling(window=14).mean()
    df_new["Rolling_21D"] = df_new["Units Sold"].rolling(window=21).mean()
    df_new["Rolling_1M"] = monthly_df_new["Units Sold"].rolling(1).mean()
    df_new["Rolling_2M"] = monthly_df_new["Units Sold"].rolling(2).mean()
    df_new["Rolling_3M"] = monthly_df_new["Units Sold"].rolling(3).mean()
    df_new["Rolling_6M"] = monthly_df_new["Units Sold"].rolling(6).mean()
    df_new["Rolling_12M"] = monthly_df_new["Units Sold"].rolling(12).mean()
    df_new["Rolling_7D_rev"]=df_new["Total Revenue"].rolling(window=7).mean()
    df_new["Rolling_1M_rev"]=monthly_df_new_rev["Total Revenue"].rolling(1).mean()

    df_new=df_new.dropna()
    if item == "Meat" or item =="Beverages" or item == "Cosmetics" or item =="Cereal":
        df_new=replace_target_encoded_features(df_new,df_ref,["Country"])
        df_new=replace_label_encoded_features(df_new,df_ref,["Sales Channel", "Order Priority","Season"])
        df_new=apply_existing_one_hot_encoding(df_new,df_ref,["Region"])
    elif item == "Baby Food" or item =="Clothes":
        df_new=replace_target_encoded_features(df_new,df_ref,["Region", "Country", "Sales Channel", "Order Priority"])

    elif item=="all" or item=="Personal Care" or item=="Vegetables" or item=="Fruits" or item=="Office Supplies":
        df_new=replace_label_encoded_features(df_new,df_ref,["Region", "Country", "Sales Channel", "Order Priority","Item Type"])
    return df_new


def replace_target_encoded_features(df_new, df_ref, features):
    for col in features:
        # Generate the encoded column name for target encoding (e.g., "Country" -> "Country_encoded")
        encoded_col = col + "_encoded"
        
        # Check if the encoded column exists in df_ref
        if encoded_col in df_ref.columns:

            # Create mapping from original to encoded value
            mapping = dict(zip(df_ref[col], df_ref[encoded_col]))
            df_new[encoded_col] = df_new[col].map(mapping)
        else:
            raise KeyError(f"Encoded column '{encoded_col}' not found in df_ref for original column '{col}'. Available columns: {df_ref.columns}")
    
    return df_new


def replace_label_encoded_features(df_new, df_ref, features):
    for col in features:
        # Normalize column names by replacing spaces with underscores
        normalized_col = re.sub(r"\s+", "_", col.strip())
        
        # Generate the encoded column name for label encoding (e.g., "Country" -> "Country_Encoded")
        encoded_col = normalized_col + "_Encoded"
        
        # Check if the encoded column exists in df_ref
        if encoded_col in df_ref.columns:

            # Create mapping from original to encoded value
            mapping = dict(zip(df_ref[col], df_ref[encoded_col]))
            df_new[encoded_col] = df_new[col].map(mapping)
        else:
            raise KeyError(f"Encoded column '{encoded_col}' not found in df_ref for original column '{col}'. Available columns: {df_ref.columns}")
    
    return df_new

def apply_existing_one_hot_encoding(df_new, df_ref, features):
    for col in features:
        # Find all one-hot encoded columns related to this feature
        one_hot_cols = [c for c in df_ref.columns if c.startswith(col + "_")]

        for one_hot_col in one_hot_cols:
            # Extract category from column name
            category = one_hot_col.split(col + "_", 1)[-1]

            # Create the same one-hot column in df_new
            df_new[one_hot_col] = (df_new[col] == category).astype(int)
    
    return df_new







# Main function
def main():
    # Streamlit UI
    st.title("Interactive Plot Selector")
    plot_type = st.selectbox("Choose a plot type:", ["Overall Yearly Sales Trend", "Overall Sales Trend Over Time", "Monthly Revenue and Profit Trends"
     ,"Monthly Sales Over Years by Order Priority", "Sales Trend by Item Type",
     "Proportion of Different Item Types","Number of Purchases by Sales Channel","Monthly Revenue by Year","Top 5 Countries by Revenue",
     "Revenue and Profit by Order Priority","Revenue by Product Category","Profit by Region","Profit Margin by Product Type",
    "Unit Price vs Units Sold", "Units Sold by Day of Month",   "Distribution of Units Sold",
    "Monthly Demand with Rolling Average and Seasonal Shading","Profit vs Units Sold by Item Type",
    "Revenue and Profit per Item Type","Total Profit by Region","Units Sold by Item Type",
    "Units Sold per Item Type","Average Units Sold per Month (Seasonality)","Sales Channel Distribution"
   ])

    show_plots(plot_type)
    st.title(" Modeling and Prediction ")
    items_list = list(df["Item Type"].unique())
    items_list.append("all")

    # Remove specific items
    items_to_remove = ["Snacks", "Household"]
    items_list = [item for item in items_list if item not in items_to_remove]
 
    item=st.selectbox("Choose Item to predict Units sold :",items_list)
    if item == "Baby Food" or item =="Clothes":
        df_enc=target_encoding(df,["Region", "Country", "Sales Channel", "Order Priority"])
    elif item == "Meat" or item =="Beverages" or item == "Cosmetics" or item =="Cereal":
        df_enc=one_hot_encoding(df,["Region"])
        df_enc=target_encoding(df_enc,[ "Country"])
        df_enc=label_encoding(df_enc,["Sales Channel", "Order Priority","Season"])



    elif item=="all" or item=="Personal Care" or item=="Vegetables" or item=="Fruits" or item=="Office Supplies" :
        df_enc=label_encoding(df,["Region", "Country", "Sales Channel", "Order Priority","Item Type"])

    if item !="all":
        show_models(item,df_enc[df_enc["Item Type"]==item])


    else:
        show_models(item,df_enc)




if __name__ == "__main__":
    main()

